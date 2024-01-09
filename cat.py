import matplotlib.pyplot as plt

# 创建一个画布
fig, ax = plt.subplots()

# 绘制草原
grass = plt.Rectangle((0, 0), 100, 50, fc='green')
ax.add_patch(grass)

# 绘制狗的身体
body = plt.Rectangle((20, 20), 20, 10, fc='brown')
ax.add_patch(body)

# 绘制狗的头部
head = plt.Circle((40, 25), 5, fc='brown')
ax.add_patch(head)

# 绘制狗的耳朵
ear1 = plt.Polygon([[45, 30], [50, 30], [50, 35]], fc='brown')
ax.add_patch(ear1)

ear2 = plt.Polygon([[45, 20], [50, 20], [50, 25]], fc='brown')
ax.add_patch(ear2)

# 绘制狗的眼睛
eye1 = plt.Circle((42, 26), 1, fc='white')
ax.add_patch(eye1)

eye2 = plt.Circle((42, 24), 1, fc='white')
ax.add_patch(eye2)

# 绘制狗的鼻子
nose = plt.Circle((44, 25), 0.5, fc='black')
ax.add_patch(nose)

# 绘制狗的嘴巴
mouth = plt.Polygon([[44, 24], [46, 24], [46, 26]], fc='black')
ax.add_patch(mouth)

# 绘制狗的尾巴
tail = plt.Polygon([[20, 25], [15, 25], [20, 30]], fc='brown')
ax.add_patch(tail)

# 设置坐标轴范围
ax.set_xlim(0, 100)
ax.set_ylim(0, 50)

# 隐藏坐标轴
ax.axis('off')

# 显示绘制结果
plt.show()

