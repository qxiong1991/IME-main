#!/usr/bin/env python
# -*- coding:utf-8 -*-  
__author__ = 'IT小叮当'
__time__ = '2021-05-24 17:00'
#!/usr/bin/env python
# -*- coding:utf-8 -*-
__author__ = 'IT小叮当'
__time__ = '2020-05-24 11:41'

import tensorflow as tf
import time

#查看是否有GPU
print('***********查看是否有GPU********************')
gpu_device_name = tf.test.gpu_device_name()
print('*********************************')
print(gpu_device_name)
print('*********************************')
time.sleep(3)
print('\n')
#查看GPU是否可用
print("@@@@@@@@@@@@@查看GPU是否可用@@@@@@@@@@@@@@")
print('@@@@@@@@@@@@@@@@@@@@@@@@@@')
print(tf.config.list_physical_devices('GPU'))
print('@@@@@@@@@@@@@@@@@@@@@@@@@@')
time.sleep(3)
print('\n')
print('#############开始进行多GPU计算测试#################')
#多GPU计算测试
# 创建计算图
c = []
for d in ['/device:GPU:0', '/device:GPU:1']:
 with tf.device(d):
   a = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[2, 3])
   b = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[3, 2])
   c.append(tf.matmul(a, b))
with tf.device('/cpu:0'):
 sum = tf.add_n(c)

print(sum)
