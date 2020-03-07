%Demonstrates compressively sampling and D-AMP recovery of an image.
clc;
clear;

addpath(genpath('..'));
addpath(genpath('../BM3D'));
%addpath(genpath('../BLS-GSM/denoising_subprograms'));
%addpath(genpath('../BLS-GSM/Added_PyrTools'));
addpath(genpath('../BM3D/BM3D-SAPCA'));

% Parameters
%denoiser1 = 'fast-BM3D';
%denoiser1 = 'Gauss';
denoiser1 = 'BM3D';

% Available options are NLM, Gauss, Bilateral, BLS-GSM, BM3D, fast-BM3D, and BM3D-SAPCA 
% denoiser2 = '';

% filename = 'house.png';
% SamplingRate = .20;
iters = 10;
% imsize = 128;

% ImIn = double(imread(filename));
% x_0 = imresize(ImIn, imsize / size(ImIn, 1));
% x_0 = ImIn;
% x_0 = x_0 / 255;
% [height, width] = size(x_0);
% n = length(x_0(:));
% m = round(n * SamplingRate);

load(['../TrainingData/StandardTestData_128Res.mat']);
squeeze_Image = squeeze(data);
permute_Image = permute(squeeze_Image, [1, 3, 2]);
reshape_Image = reshape(permute_Image, size(permute_Image, 1), size(permute_Image, 2) * size(squeeze_Image, 3));
reshape_Image = reshape_Image';

SamplingRate = 1.0;
height = 128;
width = 128;
n = int32(height * width);
m = int32(n * SamplingRate);

% Generate Gaussian Measurement Matrix
% M = randn(m, n);
% for j = 1: n
% 	M(:, j) = M(:, j) ./ sqrt(sum(abs(M(:, j)).^2));
% end

M = randn(m, n) / sqrt(double(m));
nuw = 1 / 255;
w = nuw * randn(m, 1);
index = 1;
x_0_old = reshape_Image(:, index);
% x_0 = x_0_old * 255;

n_bit = 2;
Theta = 1000.0;
% Compressively sample the image
% z = M * x_0 + w;
z = M * x_0_old + w;
[y, quan_step] = DGAMP_Quantization(z, n_bit);

% Recover Signal using D-AMP algorithms
x_hat1 = DGAMP_Iter(y, iters, height, width, nuw, n_bit, quan_step, denoiser1, M);
% x_hat1 = DAMP_1(y, iters, height, width, denoiser1, M, nuw);

% D-AMP Recovery Performance
% performance1 = PSNR(x_0, x_hat1);
%performance1 = PSNR(x_0_old, x_hat1);

% erformance2 = PSNR(x_0, x_hat2);
% [num2str(SamplingRate * 100), '% Sampling ', denoiser1, '-AMP Reconstruction PSNR = ', num2str(performance1)]
% [num2str(SamplingRate * 100), '% Sampling ', denoiser2, '-AMP Reconstruction PSNR = ', num2str(performance2)]

% Plot Recovered Signals
% subplot(1, 2, 1);
figure;
imshow(permute(reshape(x_0_old, [height, width]), [2, 1]));
title('Original Image');
% subplot(1, 2, 2);
figure;
imshow(permute(reshape(x_hat1, [height, width]), [2, 1]));

title([denoiser1, '-GAMP']);
% subplot(1,3,3);
% imshow(x_hat2 / 255);
%title([denoiser2, '-AMP']);

%x_0_old'
%x_hat1'
sqrt(mean((x_0_old - x_hat1).^2))
sqrt(sum((x_0_old - x_hat1).^2))

tmp = (double(x_0_old) - double(x_hat1)).^2;
MSE = mean(tmp(:));
psnr = -10 * log(MSE) / log(10)