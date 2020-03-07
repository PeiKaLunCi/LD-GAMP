%function [tilde_y,quan_step]=Quantization(y,n_bit,AGC_switch)
function [tilde_y, quan_step] = GAMP_Quantization(y, n_bit)
%if AGC_switch == 0
%    tilde_y=y;
%    quan_step=0;
%elseif AGC_switch==1
%y_real=real(y);
%y_imag=imag(y);
%tem_y=[y_real;y_imag];
%tem_y = y;

if n_bit == 1
    quan_step = 2;
elseif n_bit == 2 
    quan_step = 1 / 2;
else
    %quan_step=(max(abs(tem_y))+1)/2/(2^(n_bit-1));
    quan_step = 2 / 2^(n_bit);
end

DeltaTh = (0: 1 : 2^(n_bit - 1) - 1) * quan_step ;
Q_Out = (1: 2: 2^n_bit - 1) * quan_step / 2 ;

%y_R = Q_Out(end) * ( abs(y_real) > DeltaTh(end) ) ;
y_R = Q_Out(end) * (abs(y) >= DeltaTh(end));

for bIdx = length(DeltaTh): -1: 2
    %y_R = y_R + Q_Out(bIdx - 1) * ((abs(y_real) < DeltaTh(bIdx)) & (abs(y_real) >= DeltaTh(bIdx - 1)) ) ;
    y_R = y_R + Q_Out(bIdx - 1) * ((abs(y) < DeltaTh(bIdx)) & (abs(y) >= DeltaTh(bIdx - 1)) ) ;
end

%y_R = sign(y_real) .* y_R;
y_R = sign(y) .* y_R;

%y_I = Q_Out(end) * ( abs(y_imag) > DeltaTh(end) ) ;
%for bIdx = length(DeltaTh):-1:2
%    y_I = y_I + Q_Out(bIdx-1) * ( (abs(y_imag) < DeltaTh(bIdx)) & (abs(y_imag) >= DeltaTh(bIdx-1)) ) ;
%end
%y_I = sign(y_imag) .* y_I ;

%tilde_y = y_R + 1j*y_I ;    
tilde_y = y_R;
%else
%    disp('error');
%end
end