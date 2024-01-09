# TEST
import torch
from torch.backends import cudnn

x = torch.Tensor([1.0])
xx = x.cuda()
print("torch版本:", torch.__version__)
print("torch_cudatoolkit版本:", torch.version.cuda)
print("torch_cuda_可用:", torch.cuda.is_available())
print("torch_cuda_计算:", xx)
print("torch_cudnn_可用:", cudnn.is_acceptable(xx))

