import numpy as np
from scipy.io import loadmat, savemat
from matplotlib import pyplot as plt

height_img = 256
width_img = 256
channel_img = 1

data = loadmat('./kk_5_IterNum_10_hat_x.mat')
Image = data['hat_x']

Image = np.transpose(np.reshape(Image, (-1, height_img * width_img * channel_img)))
Image = np.squeeze(Image)
Image = np.reshape(Image, (height_img, width_img))

fig1 = plt.figure()
plt.imshow(np.transpose(Image), interpolation='nearest', cmap='gray')
plt.show()

test_im_name = "./TrainingData/StandardTestData_" + str(height_img) + "Res.npy"
test_images = np.load(test_im_name)
test_image = test_images[4,0,:,:]
print(test_image.shape)

fig2 = plt.figure()
plt.imshow(np.transpose(test_image), interpolation='nearest', cmap='gray')
plt.show()

x1 = Image
x2 = test_image

x_loss = np.sqrt(np.mean(np.square(x1 - x2)))
x_loss1 = np.sqrt(np.sum(np.square(x1 - x2)))

print(x1)
print(x2)
print(x_loss)
print(x_loss1)

MSE = np.mean(np.square(x1 - x2))
psnr = -10 * np.log(MSE) / np.log(10)
print(psnr)