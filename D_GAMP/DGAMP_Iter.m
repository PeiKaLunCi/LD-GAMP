function [x_hat] = DGAMP_Iter(y, iters, height, width, nuw, n_bit, quan_step, denoiser, M_func)
% function [x_hat, PSNR] = DAMP(y, iters, width, height, denoiser, M_func, Mt_func, PSNR_func)
% this function implements D-AMP based on any denoiser present in the
% denoise function
%
% Required Input:
%       y       : the measurements 
%       iters   : the number of iterations
%       width   : width of the sampled signal
%       height  : height of the sampeled signal. height=1 for 1D signals
%       denoiser: string that determines which denosier to use. e.g., 'BM3D'
%       M_func  : function handle that projects onto M. Or a matrix M.
%
% Optional Input:
%       Mt_func  : function handle that projects onto M'.
%       PSNR_func: function handle to evaluate PSNR
%
% Output:
%       x_hat   : the recovered signal.
%       PSNR    : the PSNR trajectory.

denoi = @(noisy, sigma_hat) DGAMP_Denoiser(noisy, sigma_hat, width, height, denoiser);

n = width * height;
m = length(y);

x_t = zeros(n, 1);
v_t = ones(n, 1);

s_t = zeros(m, 1);

M_sq = M_func.^2;

for ii = 1: iters
	V_t = M_sq * v_t;
	Z_t = M_func * x_t - s_t .* V_t;

	[z_tem, v_tem, s_t, tau_t] = estimator_2(y, Z_t, V_t, nuw, n_bit, quan_step);
	
	Sigma = 1 ./ (M_sq' * tau_t);
	R = x_t + Sigma .* (M_func' * s_t);
	
	% sigma_hat = sqrt(mean(abs(s_t ./ mean(tau_t)).^2));

	% sigma_hat = sqrt(mean(abs(tau_t).^2));
	vartheta = m ./ n;
	mean_hat_v = mean(v_t);
	mean_tilde_v = mean(v_tem);

	residual_z = (mean_hat_v ./ vartheta).^2 ./ (mean_hat_v ./ vartheta - mean_tilde_v) .* s_t;
	sigma_hat = sqrt(mean(abs(residual_z).^2));
	
	x_t = denoi(R, sigma_hat);
	epsilon = max(max(abs(R(:))) / 1000, 0.00001);
	% epsilon = max(abs(R(:))) / 1000 + eps;

	eta = randn(1, n);
	div = eta * ((denoi(R + epsilon .* eta', sigma_hat) - x_t) ./ epsilon) ./ n;
	v_t = div * Sigma;
	v_t = max(v_t, 0.00001);

	if sum(isnan(v_t)) > 0
		1234
		break;
	end
end
x_hat = x_t;

end

function[z_tem, v_tem, hat_s,hat_tau] = estimator_2(y, Z,V, nuw, n_bit, quan_step)

% Y = y ./ Theta;

Quan_bound = (2.^(n_bit - 1) - 1) .* quan_step;
z = Z;
v = V;

%threhold = Quan_bound + quan_step;
threhold = 2 * Quan_bound;

y_up = y + quan_step ./ 2.0;
y_low = y - quan_step ./ 2.0;
[pos1, ~] = find(y_up > Quan_bound);
[pos2, ~] = find(y_low < -Quan_bound);

% save('pos1', 'pos1');
% save('pos2', 'pos2');

y_up(pos1) = threhold;
y_low(pos2) = -threhold;

% y_up = y_up * Theta;
% y_low = y_low * Theta;

% save('y_up', 'y_up');
% save('y_low', 'y_low');

eta1 = (sign(y) .* z - min(abs(y_up), abs(y_low))) ./ sqrt((nuw * nuw + v) ./ 2.0);
eta2 = (sign(y) .* z - max(abs(y_up), abs(y_low))) ./ sqrt((nuw * nuw + v) ./ 2.0);
	
tem1 = normpdf(eta1) - normpdf(eta2);
tem2 = normcdf(eta1) - normcdf(eta2);
tem3 = eta1 .* normpdf(eta1) - eta2 .* normpdf(eta2);

Eps = 1e-15;
z_tem = z + (sign(y) .* v ./ sqrt(2 * (nuw * nuw + v))) .* (tem1 ./ (tem2 + Eps));
v_tem = v / 2 - ((v.^2) ./ (2 * (nuw * nuw + v))) .* (tem3 ./ (tem2 + Eps) + (tem1 ./ (tem2 + Eps)).^2);
	
hatz = z_tem;
hatv = v_tem;
	
hat_s = (hatz - Z) ./ V;
hat_tau = (V - hatv) ./ (V.^2);
end