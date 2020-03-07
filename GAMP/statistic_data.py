import numpy as np
import matplotlib.pyplot as plt
height_img = 256

test_im_name = "../TrainingData/StandardTestData_" + str(height_img) + "Res.npy"
test_images = np.load(test_im_name)

print(test_images.shape)

num = 50

gap = 1.0 / num

print('gap:', gap)

data = [0] * num

for i in range(num):
    if i == num - 1:
        data[i] = np.sum((gap * i) <= test_images)
    else:
        data[i] = np.sum(((gap * i) <= test_images) * (test_images < (gap * (i + 1))))

print(data)
print(sum(data))

data1 = [i / sum(data) for i in data]
print(data1)
print(sum(data1))

index = [(i + 0.0) / num  for i in range(num)]
plt.plot(index, data, 'r-', label = 'base', ms = 2)

plt.show()
"""
"""
print(test_images.max())
print(test_images.min())

num = 50

gap = 1.0 / num

print('gap:', gap)

data = [0] * num

for i in range(num):
    if i == num - 1:
        data[i] = np.sum(test_images <= (gap * (i + 1)))
    else:
        data[i] = np.sum(test_images < (gap * (i + 1)))

print(data)
print(data[-1])

data1 = [i / data[-1] for i in data]
print(data1)
print(sum(data1))

index = [(i + 0.0) / num  for i in range(num)]
plt.plot(index, data, 'r-', label = 'base', ms = 2)

plt.show()
"""

mean_iamge = np.mean(test_images)

print(mean_iamge)
var_image = np.mean(np.square(test_images - mean_iamge))
print(var_image)
"""