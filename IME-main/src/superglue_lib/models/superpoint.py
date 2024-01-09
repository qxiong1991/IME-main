# %BANNER_BEGIN%
# ---------------------------------------------------------------------
# %COPYRIGHT_BEGIN%
#
#  Magic Leap, Inc. ("COMPANY") CONFIDENTIAL
#
#  Unpublished Copyright (c) 2020
#  Magic Leap, Inc., All Rights Reserved.
#
# NOTICE:  All information contained herein is, and remains the property
# of COMPANY. The intellectual and technical concepts contained herein
# are proprietary to COMPANY and may be covered by U.S. and Foreign
# Patents, patents in process, and are protected by trade secret or
# copyright law.  Dissemination of this information or reproduction of
# this material is strictly forbidden unless prior written permission is
# obtained from COMPANY.  Access to the source code contained herein is
# hereby forbidden to anyone except current COMPANY employees, managers
# or contractors who have executed Confidentiality and Non-disclosure
# agreements explicitly covering such access.
#
# The copyright notice above does not evidence any actual or intended
# publication or disclosure  of  this source code, which includes
# information that is confidential and/or proprietary, and is a trade
# secret, of  COMPANY.   ANY REPRODUCTION, MODIFICATION, DISTRIBUTION,
# PUBLIC  PERFORMANCE, OR PUBLIC DISPLAY OF OR THROUGH USE  OF THIS
# SOURCE CODE  WITHOUT THE EXPRESS WRITTEN CONSENT OF COMPANY IS
# STRICTLY PROHIBITED, AND IN VIOLATION OF APPLICABLE LAWS AND
# INTERNATIONAL TREATIES.  THE RECEIPT OR POSSESSION OF  THIS SOURCE
# CODE AND/OR RELATED INFORMATION DOES NOT CONVEY OR IMPLY ANY RIGHTS
# TO REPRODUCE, DISCLOSE OR DISTRIBUTE ITS CONTENTS, OR TO MANUFACTURE,
# USE, OR SELL ANYTHING THAT IT  MAY DESCRIBE, IN WHOLE OR IN PART.
#
# %COPYRIGHT_END%
# ----------------------------------------------------------------------
# %AUTHORS_BEGIN%
#
#  Originating Authors: Paul-Edouard Sarlin
#
# %AUTHORS_END%
# --------------------------------------------------------------------*/
# %BANNER_END%

from pathlib import Path
import torch
from torch import nn

def simple_nms(scores, nms_radius: int):
    """ Fast Non-maximum suppression to remove nearby points """
    assert(nms_radius >= 0)

    def max_pool(x):
        return torch.nn.functional.max_pool2d(
            x, kernel_size=nms_radius*2+1, stride=1, padding=nms_radius)

    zeros = torch.zeros_like(scores)
    max_mask = scores == max_pool(scores)      
    for _ in range(2):
        supp_mask = max_pool(max_mask.float()) > 0 
        supp_scores = torch.where(supp_mask, zeros, scores)   
        new_max_mask = supp_scores == max_pool(supp_scores)   
        max_mask = max_mask | (new_max_mask & (~supp_mask))   
    return torch.where(max_mask, scores, zeros)     


def remove_borders(keypoints, scores, border: int, height: int, width: int):
    """ Removes keypoints too close to the border """
    mask_h = (keypoints[:, 0] >= border) & (keypoints[:, 0] < (height - border))
    mask_w = (keypoints[:, 1] >= border) & (keypoints[:, 1] < (width - border))
    mask = mask_h & mask_w
    return keypoints[mask], scores[mask]


def top_k_keypoints(keypoints, scores, k: int):
    if k >= len(keypoints):
        return keypoints, scores
    scores, indices = torch.topk(scores, k, dim=0)
    return keypoints[indices], scores


def sample_descriptors(keypoints, descriptors, s: int = 8):
    """ Interpolate descriptors at keypoint locations """
    b, c, h, w = descriptors.shape         
    keypoints = keypoints - s / 2 + 0.5                               
    keypoints /= torch.tensor([(w*s - s/2 - 0.5), (h*s - s/2 - 0.5)],      
                              ).to(keypoints)[None]                         
    keypoints = keypoints*2 - 1 
    args = {'align_corners': True} if torch.__version__ >= '1.3' else {}             
    descriptors = torch.nn.functional.grid_sample(
        descriptors, keypoints.view(b, 1, -1, 2), mode='bilinear', **args)     
    descriptors = torch.nn.functional.normalize(
        descriptors.reshape(b, c, -1), p=2, dim=1)                     
    return descriptors


class SuperPoint(nn.Module):
    """SuperPoint Convolutional Detector and Descriptor

    SuperPoint: Self-Supervised Interest Point Detection and
    Description. Daniel DeTone, Tomasz Malisiewicz, and Andrew
    Rabinovich. In CVPRW, 2019. https://arxiv.org/abs/1712.07629

    """
    default_config = {
        'descriptor_dim': 256,
        'nms_radius': 4,
        'keypoint_threshold': 0.005,
        'max_keypoints': -1,
        'remove_borders': 4,
    }

    def __init__(self, config):
        super().__init__()
        self.config = {**self.default_config, **config}

        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        c1, c2, c3, c4, c5 = 64, 64, 128, 128, 256

        self.conv1a = nn.Conv2d(1, c1, kernel_size=3, stride=1, padding=1)
        self.conv1b = nn.Conv2d(c1, c1, kernel_size=3, stride=1, padding=1)
        self.conv2a = nn.Conv2d(c1, c2, kernel_size=3, stride=1, padding=1)
        self.conv2b = nn.Conv2d(c2, c2, kernel_size=3, stride=1, padding=1)
        self.conv3a = nn.Conv2d(c2, c3, kernel_size=3, stride=1, padding=1)
        self.conv3b = nn.Conv2d(c3, c3, kernel_size=3, stride=1, padding=1)
        self.conv4a = nn.Conv2d(c3, c4, kernel_size=3, stride=1, padding=1)
        self.conv4b = nn.Conv2d(c4, c4, kernel_size=3, stride=1, padding=1)

        self.convPa = nn.Conv2d(c4, c5, kernel_size=3, stride=1, padding=1)
        self.convPb = nn.Conv2d(c5, 65, kernel_size=1, stride=1, padding=0)

        self.convDa = nn.Conv2d(c4, c5, kernel_size=3, stride=1, padding=1)
        self.convDb = nn.Conv2d(
            c5, self.config['descriptor_dim'],
            kernel_size=1, stride=1, padding=0)

        path = Path(__file__).parent / 'weights/superpoint_v1.pth'
        self.load_state_dict(torch.load(str(path)))

        mk = self.config['max_keypoints']
        if mk == 0 or mk < -1:
            raise ValueError('\"max_keypoints\" must be positive or \"-1\"')

        print('Loaded SuperPoint model')

    def forward(self, data):
        """ Compute keypoints, scores, descriptors for image """
        # Shared Encoder        
        x = self.relu(self.conv1a(data['image']))    #data['image']==torch.Size([1, 1, 698, 800]);[1, 64, 698, 800]到[1, 64, 698, 800]
        x = self.relu(self.conv1b(x))
        x = self.pool(x)   #[1, 64, 300, 400]
        x = self.relu(self.conv2a(x))
        x = self.relu(self.conv2b(x))
        x = self.pool(x)    #[1, 64, 150, 200]
        x = self.relu(self.conv3a(x))
        x = self.relu(self.conv3b(x))
        x = self.pool(x)             #[1, 128, 75/86/87, 100]
        x = self.relu(self.conv4a(x))
        x = self.relu(self.conv4b(x))      #[1, 128, 75/86/87, 100]
        
        # Compute the dense keypoint scores
        cPa = self.relu(self.convPa(x))  #cPa.shape=[1, 256, 75/86/87, 100]
        scores = self.convPb(cPa)     #scores.shape=[1, 65, 75/86/87, 100]
        scores = torch.nn.functional.softmax(scores, 1)[:, :-1]     #scores.shape=[1, 64, 75/86/87, 100]   [:, :-1]是什么意思               # DIM: N x 64 x H/8 x W/8 
        b, _, h, w = scores.shape
        scores = scores.permute(0, 2, 3, 1).reshape(b, h, w, 8, 8)     #[1, 75/86/87*8, 100,8,8]
        scores = scores.permute(0, 1, 3, 2, 4).reshape(b, h*8, w*8)   #[1, 75/86/87*8, 800]
        scores = simple_nms(scores, self.config['nms_radius'])            #[1, 75/86/87*8, 800]  scores是
        

        # Extract keypoints
        keypoints = [
            torch.nonzero(s > self.config['keypoint_threshold'])          #输出关键点（得分值大于0.005）的坐标
            for s in scores]
        scores = [s[tuple(k.t())] for s, k in zip(scores, keypoints)]    #返回关键点的值

        # Discard keypoints near the image borders    #移除图像边界的关键点  
        keypoints, scores = list(zip(*[
            remove_borders(k, s, self.config['remove_borders'], h*8, w*8)
            for k, s in zip(keypoints, scores)]))                            #keypoints, scores分别是关键点的坐标和分值

        # Keep the k keypoints with highest score    #选取分值最高关键点的个数k是self.config['max_keypoints']的值
        if self.config['max_keypoints'] >= 0:
            keypoints, scores = list(zip(*[
                top_k_keypoints(k, s, self.config['max_keypoints'])
                for k, s in zip(keypoints, scores)]))
        


        # Convert (h, w) to (x, y)
        keypoints = [torch.flip(k, [1]).float() for k in keypoints]       #将上面得到的关键点的坐标keypoints在一维上进行翻转，得到新的keypoints；即[679, 319]到[319., 679.]
        # Compute the dense descriptors
        cDa = self.relu(self.convDa(x))           #[1, 256, 75/86/87, 100]
        descriptors = self.convDb(cDa)            #[1, 256, 75/86/87, 100]
        descriptors = torch.nn.functional.normalize(descriptors, p=2, dim=1)
        # Extract descriptors
        descriptors = [sample_descriptors(k[None], d[None], 8)[0]          #N,C,XXXX，XXXX表示描述子在坐标下的一串向量；   keypoints=(XXX,2)和score=(XXX)是在特征图扩充后得到的坐标和分值； 描述子descriptors是在特征图大小的形式上得到的
                       for k, d in zip(keypoints, descriptors)]
        return {
            'keypoints': keypoints,
            'scores': scores,
            'descriptors': descriptors,
        }
