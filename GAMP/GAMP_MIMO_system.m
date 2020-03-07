%function   obj=MIMO_system(Input)
function obj = GAMP_MIMO_system(Input)
    %mod_size=Input.mod_size;
    N = Input.N;
    M = Input.M;
    bit = Input.bit;
    nuw = Input.nuw;
    %AGC_switch=Input.AGC_switch;

    %% Generate xo
    %obj.xo=Gen_Constellation(mod_size); % Generate original constellation sets

    %% Generate x
    %sym = modem.qammod(2^mod_size);
    %normal_scal = 1/sqrt((2/3)*(2^mod_size-1)) ;      %QAM normalization 
    %sym.input='bit';                                  %the type of input data shoud be 'bit'
    %sym.symbolorder='gray';
    %informationBit = round(rand(N*mod_size,1)) ;      %�������ж���������
    %informationSym = modulate(sym, informationBit);   %�Զ��������н���QAM����
    %x = normal_scal * reshape(informationSym,N,1) ;   %�õ���һ����QAM�����ź�
    x = Input.x;
    %% Channel
    %H=(randn(M,N)+1j*randn(M,N))/sqrt(2*N);
    H = randn(M, N) / sqrt(double(M));
    %% Noise
    %w = sqrt(nuw/2)*(randn(M,1)+1j*randn(M,1));   %������˹����
    w = nuw * randn(M, 1);   %������˹����
    
    %% Uncoded system
    z = H * x;
    y = z + w;
    %[tilde_y,quan_step]=Quantization(y,bit,AGC_switch);
    [tilde_y, quan_step] = GAMP_Quantization(y, bit);
    %% load parameters
    obj.x=x;
    obj.H=H;
    obj.z=z;
    obj.y=tilde_y;
    %obj.informationBit=informationBit;
    obj.quan_step=quan_step;
end