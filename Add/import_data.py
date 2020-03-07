import matplotlib.pyplot as plt # plt 用于显示图片
import matplotlib.image as mpimg # mpimg 用于读取图片
import numpy as np
from scipy import io

data = np.zeros(shape = (7, 1, 128, 128))

for i in range(7):
	tmp = mpimg.imread(str(i + 1) + '_gaitubao_128x128.png')
	print(tmp.shape)
	a = tmp[:,:,0]
	print(np.max(a))
	print(np.min(a))
	plt.imshow(a, interpolation = 'nearest', cmap = 'gray')
	plt.show()
	data[i, :, :, :] = a

height_img = 128
test_im_name = "../TrainingData/StandardTestData_" + str(height_img) + "Res.npy"

np.save(test_im_name, data)
io.savemat("../TrainingData/StandardTestData_" + str(height_img) + "Res.mat", {'data': data})

data1 = np.load(test_im_name)
print(data1.shape)
print(data1)
 
