clc;
clear all;

%% Parameters Setting
sampling_rate_test = 0.2;
height_img = 256;
width_img = 256;
N = int32(height_img * width_img);
M = int32(N * sampling_rate_test);
%N=128;
%M=256;
%mes = 0.8;
%AGC_switch = 1;
%TestNum = 1e2;
TestNum = 10;
%IterNum = 20;
IterNum = 10;
%modType = 'QAM';
%mod_size = 2;
bit = 8;
%bit=4;
%snr = 10;

%% Load Parameters
Input.N = N;
Input.M = M;
%Input.AGC_switch = AGC_switch;
Input.IterNum = IterNum;
%Input.modType = modType;
%Input.mod_size = mod_size;
Input.bit = bit;
%Input.mes = mes;

%GEC_MSE_Error = zeros(TestNum, IterNum);
%GER_MSE_Mean = zeros(1, IterNum);

%Input.nuw=10^(-snr/10);
%Input.nuw = 1 / 255;
Input.nuw = 0.1;

load(['../TrainingData/StandardTestData_', num2str(height_img), 'Res.mat'])
squeeze_Image = squeeze(Image);
permute_Image = permute(squeeze_Image, [1, 3, 2]);

reshape_Image = reshape(permute_Image, size(permute_Image, 1), size(permute_Image, 2) * size(permute_Image, 3));
reshape_Image = reshape_Image';

num = size(reshape_Image, 2);

TestNum = min(TestNum, num);

GAMP_MSE_Error = zeros(TestNum, IterNum);
GAMP_MSE_Mean = zeros(1, IterNum);

for kk = 1:TestNum
	if kk ~= 5		
		continue;
	end

    Input.x = reshape_Image(:,kk);
    %Input.x;
    %size(Input.x);
    Input.kk = kk;
    
    %obj=MIMO_system(Input);
    obj = GAMP_MIMO_system(Input);
    %GEC_MSE_Error(kk,:) = GEC(obj, Input);
    GAMP_MSE_Error(kk, :) = GAMP_Iter(obj, Input);
    %if mod(kk,TestNum/10)==0
    %    disp(kk/TestNum*10);
    %end
end

for ii = 1: IterNum
    %GER_MSE_Mean(ii)=mean( GEC_MSE_Error(:,ii));
    GAMP_MSE_Mean(ii) = mean(GAMP_MSE_Error(:, ii));
end

iter = 1: IterNum;
semilogy(iter, GAMP_MSE_Mean, 'LineStyle', '-', 'LineWidth', 1, 'Color', 'b', 'Marker', 'h', 'MarkerSize', 6, 'MarkerFaceColor', 'none', 'MarkerEdgeColor', 'b' );   

hold on;

xlabel('Iter');
ylabel('MSE');