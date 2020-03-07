Our implementations based on the LD-AMP, their contents can be downloaded from https://github.com/ricedsp/D-AMP_Toolbox

Primary Contents

./D_GAMP/: Matlab implementations of D-GAMP.
	DGAMP_Demo.m: Denoiser-GAMP Demo.
	DGAMP_Demo_128.m: Denoiser-GAMP Demo for 128 * 128.
	BM3D_1.m: Based on BM3D.m from the packages BM3D, but we change a litle(343 and 359 row).

./GAMP/: Matlab and python implementations of GAMP.
	statistic_data.py: Statistic the test data, in order to get the prior distribution of the test data.
	fit_prior.m: Fit the probability of the prior distribution of the test data.
	GAMP_Demo.m: GAMP Demo.
	PSNR_Calculate.py: Calculate the PSNR of GAMP.
	test.py: plot the recover picture and original picture.
	
./LD_AMP/: TensorFlow implementations of LD-AMP based on complete algorithms. 
	Train_DnCNN.py: Code to train a TensorFlow implementation of DnCNN.
	Test_DnCNN.py: Code to test a TensorFlow implementation of DnCNN.
	Train_LD_AMP.py: Code to train a TensorFlow implementation of LD-AMP based on complete algorithms.
	Test_LD_AMP.py: Code to test a TensorFlow implementation of LD-AMP based on complete algorithms.
	Train_LD_AMP_2.py: Code to train a TensorFlow implementation of LD-AMP based on complete algorithms for non-linear compress sense.
	Test_LD_AMP_2.py: Code to test a TensorFlow implementation of LD-AMP based on complete algorithms for non-linear compress sense.
	Test_DnCNN_1.py: Code to test a TensorFlow implementation of DnCNN, and the test picture's dismension is 512 * 512.
	Test_DnCNN_2.py: Code to test a TensorFlow implementation of DnCNN, in order to use for the model selection.

./LD_GAMP_D/: TensorFlow implementations of LD-GAMP based on DnCNN. 
	Train_LD_GAMP_D.py: Code to train a TensorFlow implementation of LD-GAMP based on DnCNN.
	Test_LD_GAMP_D.py: Code to test a TensorFlow implementation of LD-GAMP based on DnCNN.

./LD_GAMP_R/: TensorFlow implementations of Learned D-GAMP based on ResNet.
	Train_ResNet.py: Code to train a TensorFlow implementation of ResNet.
	Test_ResNet.py: Code to test a TensorFlow implementation of ResNet.
	Train_LD_GAMP_R.py: Code to train a TensorFlow implementation of LD-GAMP based on ResNet.
	Test_LD_GAMP_R.py: Code to test a TensorFlow implementation of LD-GAMP based on ResNet.
	Test_LD_GAMP_R_128.py: Code to test a TensorFlow implementation of LD-GAMP based on ResNet for 128 * 128.
	Test_ResNet_1.py: Code to test a TensorFlow implementation of ResNet, and the test picture's dismension is 512 * 512.
	Test_ResNet_2.py: Code to test a TensorFlow implementation of ResNet, in order to use for the model selection.

./Verification_Of_Hypothesis/: python mplementations of model selection.
	Verification_Of_Hypothesis.py: python mplementations of model selection.

Data:

./TrainingData/: The TrainingData is empty directories on github, which can be downloaded from https://rice.app.box.com/v/LDAMP-LargeFiles.
./BM3D_images/: Download the BM3D images from http://www.cs.tut.fi/~foi/GCF-BM3D/BM3D_images.zip

Packages:

./BM3D/: This download includes the BM3D. The latest versions of the package can be found at: BM3D: http://www.cs.tut.fi/~foi/GCF-BM3D/