import numpy as np

## Network Parameters
height_img = 40
width_img = 40
channel_img = 1

n_Val_Images = 10000 # Must be less than 21504

#test_im_name = "/home/CHEN_QUN/Work/D-AMP_Toolbox-master/LDAMP_TensorFlow/TrainingData/ValidationData_patch" + str(height_img) + ".npy"
test_im_name = "../TrainingData/ValidationData_patch" + str(
    height_img) + ".npy"

test_images = np.load(test_im_name)
test_images = test_images[:, 0, :, :]

test_images = test_images[: n_Val_Images, :, :]
print('test_images.shape:', test_images.shape)

batch_x_test = np.transpose(np.reshape(test_images, (-1, height_img * width_img * channel_img)))
print('batch_x_test.shape:', batch_x_test.shape)

#EDNet_Reconstructing = np.load('EDNet_Reconstructing.npy')
#print('EDNet_Reconstructing.shape:', EDNet_Reconstructing.shape)

DnCNN_Reconstructing = np.load('../LDAMP/DnCNN_Reconstructing.npy')
print('DnCNN_Reconstructing.shape:', DnCNN_Reconstructing.shape)

sResNet_Reconstructing = np.load('../LDGAMP_sResNet/sResNet_Reconstructing.npy')
print('sResNet_Reconstructing.shape:', sResNet_Reconstructing.shape)

#DenseNet_Reconstructing = np.load('DenseNet_Reconstructing.npy')
#print('DenseNet_Reconstructing.shape:', DenseNet_Reconstructing.shape)

print('batch_x_test:', batch_x_test)
#print(EDNet_Reconstructing)
print('DnCNN_Reconstructing:', DnCNN_Reconstructing)
print('sResNet_Reconstructing:', sResNet_Reconstructing)

#print(DenseNet_Reconstructing)


#EDNet_varepsilon = batch_x_test - EDNet_Reconstructing
DnCNN_varepsilon = batch_x_test - DnCNN_Reconstructing
sResNet_varepsilon = batch_x_test - sResNet_Reconstructing
#DenseNet_varepsilon = batch_x_test - DenseNet_Reconstructing

#print('EDNet_varepsilon.shape:', EDNet_varepsilon.shape)
print('DnCNN_varepsilon.shape:', DnCNN_varepsilon.shape)
print('sResNet_varepsilon.shape:', sResNet_varepsilon.shape)
#print('DenseNet_varepsilon.shape:', DenseNet_varepsilon.shape)

#EDNet_T = np.mean(EDNet_varepsilon) / np.sqrt(np.mean(np.square(EDNet_varepsilon)) / (n_Val_Images * height_img * width_img * channel_img))
DnCNN_T = np.mean(DnCNN_varepsilon) / np.sqrt(np.mean(np.square(DnCNN_varepsilon)) / (n_Val_Images * height_img * width_img * channel_img))
sResNet_T = np.mean(sResNet_varepsilon) / np.sqrt(np.mean(np.square(sResNet_varepsilon)) / (n_Val_Images * height_img * width_img * channel_img))
#DenseNet_T = np.mean(DenseNet_varepsilon) / np.sqrt(np.mean(np.square(DenseNet_varepsilon)) / (n_Val_Images * height_img * width_img * channel_img))

#print('EDNet_T.shape:', EDNet_T.shape)
print('DnCNN_T.shape:', DnCNN_T.shape)
print('sResNet_T.shape:', sResNet_T.shape)
#print('DenseNet_T.shape:', DenseNet_T.shape)

#print('EDNet_T:', EDNet_T)
print('DnCNN_T:', DnCNN_T)
print('sResNet_T:', sResNet_T)
#print('DenseNet_T:', DenseNet_T)

#EDNet_psnr = -10 * np.log(np.mean(np.square(batch_x_test - EDNet_Reconstructing))) / np.log(10)
DnCNN_psnr = -10 * np.log(np.mean(np.square(batch_x_test - DnCNN_Reconstructing))) / np.log(10)
sResNet_psnr = -10 * np.log(np.mean(np.square(batch_x_test - sResNet_Reconstructing))) / np.log(10)
#DenseNet_psnr = -10 * np.log(np.mean(np.square(batch_x_test - DenseNet_Reconstructing))) / np.log(10)

#print('EDNet_psnr.shape:', EDNet_psnr.shape)
print('DnCNN_psnr.shape:', DnCNN_psnr.shape)
print('sResNet_psnr.shape:', sResNet_psnr.shape)
#print('DenseNet_psnr.shape:', DenseNet_psnr.shape)

#print('EDNet_psnr:', EDNet_psnr)
print('DnCNN_psnr:', DnCNN_psnr)
print('sResNet_psnr:', sResNet_psnr)
#print('DenseNet_psnr:', DenseNet_psnr)

# ------------------------------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------------------------

#print('np.mean(EDNet_varepsilon - np.mean(EDNet_varepsilon)):', np.mean(EDNet_varepsilon - np.mean(EDNet_varepsilon)))
print('np.mean(DnCNN_varepsilon - np.mean(DnCNN_varepsilon)):', np.mean(DnCNN_varepsilon - np.mean(DnCNN_varepsilon)))
print('np.mean(sResNet_varepsilon - np.mean(sResNet_varepsilon)):', np.mean(sResNet_varepsilon - np.mean(sResNet_varepsilon)))
#print('np.mean(DenseNet_varepsilon - np.mean(DenseNet_varepsilon)):', np.mean(DenseNet_varepsilon - np.mean(DenseNet_varepsilon)))

#e_EDNet = np.square(EDNet_varepsilon - np.mean(EDNet_varepsilon))
e_DnCNN = np.square(DnCNN_varepsilon - np.mean(DnCNN_varepsilon))
e_sResNet = np.square(sResNet_varepsilon - np.mean(sResNet_varepsilon))
#e_DenseNet = np.square(DenseNet_varepsilon - np.mean(DenseNet_varepsilon))

#print('e_EDNet.shape:', e_EDNet.shape)
print('e_DnCNN.shape:', e_DnCNN.shape)
print('e_sResNet.shape:', e_sResNet.shape)
#print('e_DenseNet.shape:', e_DenseNet.shape)

#mean_e_EDNet = np.mean(e_EDNet)
#s_e_EDNet = np.sum(np.square(e_EDNet - mean_e_EDNet)) / (n_Val_Images * height_img * width_img * channel_img - 1.0)

mean_e_DnCNN = np.mean(e_DnCNN)
s_e_DnCNN = np.sum(np.square(e_DnCNN - mean_e_DnCNN)) / (n_Val_Images * height_img * width_img * channel_img - 1.0)

mean_e_sResNet = np.mean(e_sResNet)
s_e_sResNet = np.sum(np.square(e_sResNet - mean_e_sResNet)) / (n_Val_Images * height_img * width_img * channel_img - 1.0)

#mean_e_DenseNet = np.mean(e_DenseNet)
#s_e_DenseNet = np.sum(np.square(e_DenseNet - mean_e_DenseNet)) / (n_Val_Images * height_img * width_img * channel_img - 1.0)

#print('mean_e_EDNet.shape:', mean_e_EDNet.shape)
print('mean_e_DnCNN.shape:', mean_e_DnCNN.shape)
print('mean_e_sResNet.shape:', mean_e_sResNet.shape)
#print('mean_e_DenseNet.shape:', mean_e_DenseNet.shape)

#print('s_e_EDNet.shape:', s_e_EDNet.shape)
print('s_e_DnCNN.shape:', s_e_DnCNN.shape)
print('s_e_sResNet.shape:', s_e_sResNet.shape)
#print('s_e_DenseNet.shape:', s_e_DenseNet.shape)

#print('mean_e_EDNet:', mean_e_EDNet)
print('mean_e_DnCNN:', mean_e_DnCNN)
print('mean_e_sResNet:', mean_e_sResNet)
#print('mean_e_DenseNet:', mean_e_DenseNet)

#print('s_e_EDNet:', s_e_EDNet)
print('s_e_DnCNN:', s_e_DnCNN)
print('s_e_sResNet:', s_e_sResNet)
#print('s_e_DenseNet:', s_e_DenseNet)

#T_EDNet_DnCNN = (mean_e_EDNet - mean_e_DnCNN) / np.sqrt((s_e_EDNet + s_e_DnCNN) / (n_Val_Images * height_img * width_img * channel_img))
T_sResNetNet_DnCNN = (mean_e_sResNet - mean_e_DnCNN) / np.sqrt((s_e_sResNet + s_e_DnCNN) / (n_Val_Images * height_img * width_img * channel_img))
#T_DnCNN_DenseNet = (mean_e_DnCNN - mean_e_DenseNet) / np.sqrt((s_e_DnCNN + s_e_DenseNet) / (n_Val_Images * height_img * width_img * channel_img))
#T_EDNet_DenseNet = (mean_e_EDNet - mean_e_DenseNet) / np.sqrt((s_e_EDNet + s_e_DenseNet) / (n_Val_Images * height_img * width_img * channel_img))

#print('T_EDNet_DnCNN.shape:', T_EDNet_DnCNN.shape)
print('T_sResNetNet_DnCNN.shape:', T_sResNetNet_DnCNN.shape)
#print('T_DnCNN_DenseNet.shape:', T_DnCNN_DenseNet.shape)
#print('T_EDNet_DenseNet.shape:', T_EDNet_DenseNet.shape)

#print('T_EDNet_DnCNN:', T_EDNet_DnCNN)
print('T_sResNetNet_DnCNN:', T_sResNetNet_DnCNN)
#print('T_DnCNN_DenseNet:', T_DnCNN_DenseNet)
#print('T_EDNet_DenseNet:', T_EDNet_DenseNet)
