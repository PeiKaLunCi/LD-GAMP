function [ denoised ] = DGAMP_Denoiser(noisy, sigma_hat, width, height, denoiser)
% function [ denoised ] = denoise(noisy,sigma_hat,width,height,denoiser)
% DENOISE takes a signal with additive white Guassian noisy and an estimate
% of the standard deviation of that noise and applies some denosier to
% produce a denoised version of the input signal
% Input:
%       noisy       : signal to be denoised
%       sigma_hat   : estimate of the standard deviation of the noise
%       width   : width of the noisy signal
%       height  : height of the noisy signal. height=1 for 1D signals
%       denoiser: string that determines which denosier to use. e.g.
%       denoiser='BM3D'
% Output:
%       denoised   : the denoised signal.

% To apply additional denoisers within the D-AMP framework simply add
% aditional case statements to this function and modify the calls to D-AMP

noisy=reshape(noisy,[width,height]);

switch denoiser
    case 'BM3D'
        %noisy = noisy * 255;
        [NA, output] = BM3D_1(1, noisy, sigma_hat, 'np', 0);
        %output=255*output;
    case 'fast-BM3D'
        noisy=real(noisy);
        [NA, output]=BM3D_1(1,noisy,sigma_hat,'lc',0);
        %output=255*output;
    case 'BM3D-SAPCA'
        %output = 255*BM3DSAPCA2009((1/255)*noisy,(1/255)*sigma_hat);
        output = BM3DSAPCA2009(noisy, sigma_hat);
    otherwise
        error('Unrecognized Denoiser')
end
    denoised=output(:);
end

