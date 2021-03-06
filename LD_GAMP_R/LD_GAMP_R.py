import numpy as np
import tensorflow as tf

def SetNetworkParams(new_height_img, new_width_img,new_channel_img, new_filter_height,new_filter_width,\
                     new_num_filters,new_n_GAMP_layers, new_sampling_rate,\
                     new_BATCH_SIZE,new_sigma_w,new_n,new_m,new_training,use_adaptive_weights=False):
    global num_layers_in_recon_block, height_img, width_img, channel_img, filter_height, filter_width, num_filters, n_GAMP_layers,\
        sampling_rate, BATCH_SIZE, sigma_w, n, m, n_fp, m_fp, is_complex, training, adaptive_weights
    ##############################################

    height_img = new_height_img
    width_img = new_width_img
    channel_img = new_channel_img
    filter_height = new_filter_height
    filter_width = new_filter_width
    num_filters = new_num_filters
    n_GAMP_layers = new_n_GAMP_layers
    sampling_rate = new_sampling_rate
    BATCH_SIZE = new_BATCH_SIZE
    sigma_w = new_sigma_w
    n = new_n
    m = new_m

    n_fp = np.float32(n)
    m_fp = np.float32(m)
    is_complex=False#Just the default
    adaptive_weights=use_adaptive_weights
    training=new_training

def ListNetworkParameters():
    ##############################################

    print('height_img = ', height_img)
    print('width_img = ', width_img)
    print('channel_img = ', channel_img)
    print('filter_height = ', filter_height)
    print('filter_width = ', filter_width)
    print('num_filters = ', num_filters)
    print('n_GAMP_layers = ', n_GAMP_layers)
    print('sampling_rate = ', sampling_rate)
    print('BATCH_SIZE = ', BATCH_SIZE)
    print('sigma_w = ', sigma_w)
    print('n = ', n)
    print('m = ', m)

#Form the measurement operators
def GenerateMeasurementOperators(mode):
    global is_complex
    if mode=='gaussian':
        is_complex=False
        A_val = np.float32(1. / np.sqrt(m_fp) * np.random.randn(m, n))# values that parameterize the measurement model. This could be the measurement matrix itself or the random mask with coded diffraction patterns.
        #A_val = np.float32(1. / np.sqrt(n_fp) * np.random.randn(m, n))

        A_val_tf = tf.placeholder(tf.float32, [m, n])  #A placeholer is used so that the large matrix isn't put into the TF graph (2GB limit)

        def A_handle(A_vals_tf,x):
            return tf.matmul(A_vals_tf,x)

        def At_handle(A_vals_tf,z):
            return tf.matmul(A_vals_tf,z,adjoint_a=True)
    elif mode == 'complex-gaussian':
            is_complex = True
            A_val = np.complex64(1/np.sqrt(2.)*((1. / np.sqrt(m_fp) * np.random.randn(m,n))+1j*(1. / np.sqrt(m_fp) * np.random.randn(m,n))))
            y_measured = tf.placeholder(tf.complex64, [m, None])
            A_val_tf = tf.placeholder(tf.complex64, [m, n])  # A placeholer is used so that the large matrix isn't put into the TF graph (2GB limit)

            def A_handle(A_vals_tf, x):
                return tf.matmul(A_vals_tf, tf.complex(x,tf.zeros([n,BATCH_SIZE],dtype=tf.float32)))

            def At_handle(A_vals_tf, z):
                return tf.matmul(A_vals_tf, z, adjoint_a=True)
    elif mode=='coded-diffraction':
        is_complex=True
        A_val = np.zeros([n, 1]) + 1j * np.zeros([n, 1])
        A_val[0:n] = np.exp(1j*2*np.pi*np.random.rand(n,1))#The random sign vector

        global sparse_sampling_matrix
        rand_col_inds=np.random.permutation(range(n))
        rand_col_inds=rand_col_inds[0:m]
        row_inds = range(m)
        inds=zip(row_inds,rand_col_inds)
        vals=tf.ones(m, dtype=tf.complex64);
        sparse_sampling_matrix = tf.SparseTensor(indices=inds, values=vals, dense_shape=[m,n])

        A_val_tf = tf.placeholder(tf.complex64, [n, 1])
        def A_handle(A_val_tf, x):
            sign_vec = A_val_tf[0:n]
            signed_x = tf.multiply(sign_vec, tf.complex(x,tf.zeros([n,BATCH_SIZE],dtype=tf.float32)))
            signed_x = tf.reshape(signed_x, [height_img, width_img, BATCH_SIZE])
            signed_x=tf.transpose(signed_x)#Transpose because fft2d operates upon the last two axes
            F_signed_x = tf.fft2d(signed_x)
            F_signed_x=tf.transpose(F_signed_x)
            F_signed_x = tf.reshape(F_signed_x, [height_img * width_img, BATCH_SIZE])*1./np.sqrt(m_fp)#This is a different normalization than in Matlab because the FFT is implemented differently in Matlab
            out = tf.sparse_tensor_dense_matmul(sparse_sampling_matrix,F_signed_x,adjoint_a=False)
            return out

        def At_handle(A_val_tf, z):
            sign_vec=A_val_tf[0:n]
            z_padded = tf.sparse_tensor_dense_matmul(sparse_sampling_matrix,z,adjoint_a=True)
            z_padded = tf.reshape(z_padded, [height_img, width_img, BATCH_SIZE])
            z_padded=tf.transpose(z_padded)#Transpose because fft2d operates upon the last two axes
            Finv_z = tf.ifft2d(z_padded)
            Finv_z = tf.transpose(Finv_z)
            Finv_z = tf.reshape(Finv_z, [height_img*width_img, BATCH_SIZE])
            out = tf.multiply(tf.conj(sign_vec), Finv_z)*n_fp/np.sqrt(m)
            return out
    elif mode=='Fast-JL':#Measurement matrix close to a fast JL transform. True fast JL would use hadamard transform and a sparse sampling matrix with multiple nz elements per row
        is_complex=False
        A_val = np.zeros([n, 1])
        A_val[0:n] = np.sign(2*np.random.rand(n,1)-1)

        global sparse_sampling_matrix
        rand_col_inds=np.random.permutation(range(n))
        rand_col_inds=rand_col_inds[0:m]
        row_inds = range(m)
        inds=zip(row_inds,rand_col_inds)
        vals=tf.ones(m, dtype=tf.float32);
        sparse_sampling_matrix = tf.SparseTensor(indices=inds, values=vals, dense_shape=[m,n])

        A_val_tf = tf.placeholder(tf.float32, [n, 1])
        def A_handle(A_val_tf, x):
            sign_vec = A_val_tf[0:n]
            signed_x = tf.multiply(sign_vec, x)
            signed_x = tf.reshape(signed_x, [height_img*width_img, BATCH_SIZE])
            signed_x=tf.transpose(signed_x)#Transpose because dct operates upon the last axes
            F_signed_x = mydct(signed_x, type=2, norm='ortho')
            F_signed_x=tf.transpose(F_signed_x)
            F_signed_x = tf.reshape(F_signed_x, [height_img * width_img, BATCH_SIZE])*np.sqrt(n_fp/m_fp)
            out = tf.sparse_tensor_dense_matmul(sparse_sampling_matrix,F_signed_x,adjoint_a=False)
            return out

        def At_handle(A_val_tf, z):
            sign_vec=A_val_tf[0:n]
            z_padded = tf.sparse_tensor_dense_matmul(sparse_sampling_matrix,z,adjoint_a=True)
            z_padded = tf.reshape(z_padded, [height_img*width_img, BATCH_SIZE])
            z_padded=tf.transpose(z_padded)#Transpose because dct operates upon the last axes
            Finv_z = myidct(z_padded,type=2,norm='ortho')
            Finv_z = tf.transpose(Finv_z)
            Finv_z = tf.reshape(Finv_z, [height_img*width_img, BATCH_SIZE])
            out = tf.multiply(sign_vec, Finv_z)*np.sqrt(n_fp/m_fp)
            return out
    else:
        raise ValueError('Measurement mode not recognized')
    return [A_handle, At_handle, A_val, A_val_tf]

def mydct(x,type=2,norm='ortho'):
    assert type==2 and norm=='ortho', 'Currently only type-II orthonormalized DCTs are supported'
    assert BATCH_SIZE == 1, 'Fast-JL measurement matrices currently only support batch sizes of one'
    #https://antimatter15.com/2015/05/cooley-tukey-fft-dct-idct-in-under-1k-of-javascript/
    y=tf.concat([x,tf.zeros([1,n],tf.float32)],axis=1)
    Y=tf.fft(tf.complex(y,tf.zeros([1,2*n],tf.float32)))
    Y=Y[:,:n]
    k = tf.complex(tf.range(n, dtype=tf.float32), tf.zeros(n, dtype=tf.float32))
    Y*=tf.exp(-1j*np.pi*k/(2.*n_fp))
    return tf.real(Y)/tf.sqrt(n_fp)*tf.sqrt(2.)
    # return tf.spectral.dct(x,type=2,norm='ortho')

def myidct(X,type=2,norm='ortho'):
    assert type==2 and norm=='ortho', 'Currently only type-II orthonormalized DCTs are supported'
    assert BATCH_SIZE==1, 'Fast-JL measurement matrices currently only support batch sizes of one'
    #https://antimatter15.com/2015/05/cooley-tukey-fft-dct-idct-in-under-1k-of-javascript/
    temp0=tf.reverse(X,[-1])
    temp1=tf.manip.roll(temp0,shift=1,axis=1)
    temp2=temp1[:,1:]
    temp3=tf.pad(temp2,[[0,0],[1,0]],"CONSTANT")
    Z=tf.complex(X,-temp3)
    k = tf.complex(tf.range(n,dtype=tf.float32),tf.zeros(n,dtype=tf.float32))
    Z*=tf.exp(1j*np.pi*k/(2.*n_fp))
    temp4=tf.real(tf.ifft(Z))
    even_new=temp4[:,0:n/2]
    odd_new=tf.reverse(temp4[:,n/2:],[-1])
    #https://stackoverflow.com/questions/44952886/tensorflow-merge-two-2-d-tensors-according-to-even-and-odd-indices
    x=tf.reshape(
        tf.transpose(tf.concat([even_new, odd_new], axis=0)),
        [1,n])
    return tf.real(x)*tf.sqrt(n_fp)*1/tf.sqrt(2.)

def mydct_np(x,type=2,norm='ortho'):
    assert type==2 and norm=='ortho', 'Currently only type-II orthonormalized DCTs are supported'
    assert BATCH_SIZE == 1, 'Fast-JL measurement matrices currently only support batch sizes of one'
    #https://antimatter15.com/2015/05/cooley-tukey-fft-dct-idct-in-under-1k-of-javascript/
    N=len(x)
    y=np.zeros(2*N)
    y[:N]=x
    Y=np.fft.fft(y)[:N]
    k=np.float32(range(N))
    Y*=np.exp(-1j*np.pi*k/(2*N))/np.sqrt(N/2.)
    return Y.real

def myidct_np(X,type=2,norm='ortho'):
    assert type==2 and norm=='ortho', 'Currently only type-II orthonormalized DCTs are supported'
    assert BATCH_SIZE == 1, 'Fast-JL measurement matrices currently only support batch sizes of one'
    #https://antimatter15.com/2015/05/cooley-tukey-fft-dct-idct-in-under-1k-of-javascript/
    N=len(X)
    Z=X-1j*np.append([0.],np.flip(X,0)[:N-1])
    k = np.float32(range(N))
    Z*=np.exp(1j*np.pi*k/(2*N))
    temp=np.real(np.fft.ifft(Z))
    x=np.zeros(X.size)
    even_new= temp[0:N/2]
    odd_new=np.flip(temp[N/2:],0)
    x[0::2] =even_new
    x[1::2]=odd_new
    return np.real(x)*np.sqrt(N/2.)

def GenerateMeasurementMatrix(mode):
    if mode == 'gaussian':
        A_val = np.float32(1. / np.sqrt(m_fp) * np.random.randn(m, n))
        #A_val = np.float32(1. / np.sqrt(n_fp) * np.random.randn(m, n))

    elif mode=='complex-gaussian':
        A_val = np.complex64(1 / np.sqrt(2.) * ((1. / np.sqrt(m_fp) * np.random.randn(m, n)) + 1j * (1. / np.sqrt(m_fp) * np.random.randn(m, n))))
    elif mode=='coded-diffraction':
        A_val = np.zeros([n, 1]) + 1j * np.zeros([n, 1])
        A_val[0:n] = np.exp(1j*2*np.pi*np.random.rand(n,1))#The random sign vector
        rand_col_inds=np.random.permutation(range(n))
        rand_col_inds=rand_col_inds[0:m]
        row_inds = range(m)
        inds=zip(row_inds,rand_col_inds)
        vals=tf.ones(m, dtype=tf.complex64);
        sparse_sampling_matrix = tf.SparseTensor(indices=inds, values=vals, dense_shape=[m,n])
    elif mode == 'Fast-JL':
        A_val = np.zeros([n, 1])
        A_val[0:n] = np.sign(2*np.random.rand(n,1)-1)
        rand_col_inds=np.random.permutation(range(n))
        rand_col_inds=rand_col_inds[0:m]
        row_inds = range(m)
        inds=zip(row_inds,rand_col_inds)
        vals=tf.ones(m, dtype=tf.float32);
        sparse_sampling_matrix = tf.SparseTensor(indices=inds, values=vals, dense_shape=[m,n])
    else:
        raise ValueError('Measurement mode not recognized')
    return A_val

def GenerateMeasurementMatrix_Ex(mode):
    if mode == 'gaussian':
        #A_val = np.float32(1. / np.sqrt(m_fp) * np.random.randn(m, n))
        #A_val = np.float32(1. / np.sqrt(n_fp) * np.random.randn(m, n))
        A_val = np.eye(m, n, dtype = np.float32)
    elif mode=='complex-gaussian':
        A_val = np.complex64(1 / np.sqrt(2.) * ((1. / np.sqrt(m_fp) * np.random.randn(m, n)) + 1j * (1. / np.sqrt(m_fp) * np.random.randn(m, n))))
    elif mode=='coded-diffraction':
        A_val = np.zeros([n, 1]) + 1j * np.zeros([n, 1])
        A_val[0:n] = np.exp(1j*2*np.pi*np.random.rand(n,1))#The random sign vector
        rand_col_inds=np.random.permutation(range(n))
        rand_col_inds=rand_col_inds[0:m]
        row_inds = range(m)
        inds=zip(row_inds,rand_col_inds)
        vals=tf.ones(m, dtype=tf.complex64);
        sparse_sampling_matrix = tf.SparseTensor(indices=inds, values=vals, dense_shape=[m,n])
    elif mode == 'Fast-JL':
        A_val = np.zeros([n, 1])
        A_val[0:n] = np.sign(2*np.random.rand(n,1)-1)
        rand_col_inds=np.random.permutation(range(n))
        rand_col_inds=rand_col_inds[0:m]
        row_inds = range(m)
        inds=zip(row_inds,rand_col_inds)
        vals=tf.ones(m, dtype=tf.float32);
        sparse_sampling_matrix = tf.SparseTensor(indices=inds, values=vals, dense_shape=[m,n])
    else:
        raise ValueError('Measurement mode not recognized')
    return A_val

def Expection_Variance(y, Z, V, quan_step, n_bit, sigma_w, dist):
    Quan_bound = (np.power(2.0, n_bit - 1.0) - 1.0) * quan_step
    print('Quan_bound:', Quan_bound)
    y_up = y + quan_step / 2.0
    print('y_up:', y_up)
    y_low = y - quan_step / 2.0
    print('y_low:', y_low)

    #threhold = 2 * Quan_bound
    #threhold = 5 * Quan_bound
    #threhold = 4 * Quan_bound
    threhold = 10 * Quan_bound
    print('threhold:', threhold)

    condition1 = tf.cast(x = tf.greater(x = y_up, y = Quan_bound), dtype = tf.float32)
    print('condition1:', condition1)
    y_up = (1.0 - condition1) * y_up + condition1 * threhold
    print('y_up:', y_up)

    condition2 = tf.cast(x = tf.less(x = y_low, y = -Quan_bound), dtype = tf.float32)
    print('condition2:', condition2)
    y_low = (1.0 - condition2) * y_low + condition2 * (-threhold)
    print('y_low:', y_low)

    eta1 = (tf.sign(x = y) * Z - tf.minimum(x = tf.abs(x = y_up), y = tf.abs(x = y_low))) / tf.sqrt(x = (sigma_w * sigma_w + V) / 2.0)
    print('eta1:', eta1)
    eta2 = (tf.sign(x = y) * Z - tf.maximum(x = tf.abs(x = y_up), y = tf.abs(x = y_low))) / tf.sqrt(x = (sigma_w * sigma_w + V) / 2.0)
    print('eta2:', eta2)

    #condition3 = tf.cast(x = tf.less(x = eta2, y = -100), dtype = tf.float32)
    #print('condition3:', condition3)

    #tmp1 = dist.prob(eta1) - (1.0 - condition3) * dist.prob(eta2)
    tmp1 = dist.prob(eta1) - dist.prob(eta2)
    print('tmp1:', tmp1)

    #tmp2 = eta1 * dist.prob(eta1) - (1.0 - condition3) * eta2 * dist.prob(eta2)
    tmp2 = dist.cdf(eta1) - dist.cdf(eta2)
    print('tmp2:', tmp2)

    #tmp3 = dist.cdf(eta1) - (1.0 - condition3) * dist.cdf(eta2)
    tmp3 = eta1 * dist.prob(eta1) - eta2 * dist.prob(eta2)
    print('tmp3:', tmp3)

    eps = 1e-15
    #eps = 0.0
    ztem = Z + (tf.sign(x = y) * V / tf.sqrt(x = 2.0 * (sigma_w * sigma_w + V))) * (tmp1 / (tmp2 + eps))
    print('ztem:', ztem)

    vtem = V / 2.0 - (tf.square(x = V) / (2.0 * (sigma_w * sigma_w + V))) * (tmp3 / (tmp2 + eps) + tf.square(x = tmp1 / (tmp2 + eps)))
    print('vtem:', vtem)

    # return ztem, vtem
    return ztem, vtem, y_up, y_low, eta1, eta2, tmp1, tmp2, tmp3

#Learned DGAMP
def LDGAMP_ResNet(y,A_handle,At_handle,A_val,theta,x_true, sigma_w, quan_step, n_bit, tie,training=False,LayerbyLayer=True):
    print('------------------------------- LDGAMP_ResNet -------------------------------')
    s = tf.zeros(shape = (m, BATCH_SIZE), dtype = tf.float32)
    #s = y
    print('s:', s)


    #print('BATCH_SIZE:', BATCH_SIZE)
    #mhat_one =  tf.sqrt(tf.abs(tf.trace(tf.matmul(y, y, transpose_b = True)) / (BATCH_SIZE * n) - m_fp / n_fp * sigma_w * sigma_w))
    #mhat_old = mhat_one * tf.ones(shape = (n, 1), dtype = tf.float32)
    #print('mhat_old:', mhat_old)

    #mhat = mhat_one * tf.ones_like(x_true)
    #mhat = tf.maximum(mhat, 0.001)
    mhat = tf.zeros_like(x_true)
    print('mhat:', mhat)

    #if 1 == BATCH_SIZE:
    #    vhat = tf.ones_like(tensor = mhat, dtype = tf.float32)
    #    print('vhat:', vhat)
    #else:
    #    vhat_old = tf.reduce_mean(input_tensor = tf.square(x = x_true - mhat_old), axis = 1, keepdims = True)
    #    print('vhat_old:', vhat_old)
    #    vhat = tf.tile(input = vhat_old, multiples = [1, BATCH_SIZE])
    vhat = tf.ones_like(tensor=mhat, dtype=tf.float32)
    #vhat_one = (tf.reduce_sum(tf.square(y - tf.matmul(A_val, mhat))) - m_fp * sigma_w * sigma_w) / (tf.trace(tf.matmul(A_val, A_val, transpose_b = True)))
    #vhat = vhat_one * tf.ones_like(tensor=mhat, dtype=tf.float32)
    #vhat = tf.maximum(vhat, 0.001)
    print('vhat:', vhat)

    s_list = []
    s_list.append(s)

    mhat_list = []
    mhat_list.append(mhat)

    vhat_list = []
    vhat_list.append(vhat)

    V_list = []
    Z_list = []
    ztem_list = []
    vtem_list = []

    t_list = []
    Sigma_list = []
    R_list = []

    layers_list = []

    A_2 = A_val * A_val

    add_w = 0.0
    beta = 0.8

    dist = tf.contrib.distributions.Normal(0.0, 1.0)

    for iter in range(n_GAMP_layers):
        print("iter:", iter)
        if is_complex:
            r = tf.complex(xhat,tf.zeros([n,BATCH_SIZE],dtype=tf.float32)) + At_handle(A_val,z)
            rvar = (1. / m_fp * tf.reduce_sum(tf.square(tf.abs(z)),axis=0))#In the latest version of TF, abs can handle complex values
        else:
            V = tf.matmul(a = A_2, b = vhat_list[iter])
            #V = tf.maximum(x = V, y = 0.00001)
            print('V:', V)
            V_list.append(V)

            Z = tf.matmul(a = A_val, b = mhat_list[iter]) - s_list[iter] * V_list[iter]
            #Z = tf.maximum(x = Z, y = 0.00001)

            print('Z:', Z)
            Z_list.append(Z)
            print('------------------------------- Expection_Variance -------------------------------')
            #ztem, vtem = Expection_Variance(y = y, Z = Z_list[iter], V = V_list[iter], quan_step = quan_step, n_bit = n_bit, sigma_w = sigma_w, dist = dist)
            ztem, vtem, y_up, y_low, eta1, eta2, tmp1, tmp2, tmp3 = \
                Expection_Variance(y=y, Z=Z_list[iter], V=V_list[iter], quan_step=quan_step, n_bit=n_bit, sigma_w=sigma_w, dist=dist)
            #z_hat = tf.maximum(x = z_hat, y = 0.00001)
            #v_hat = tf.maximum(x = v_hat, y = 0.00001)

            print('------------------------------- Expection_Variance -------------------------------')

            print('ztem:', ztem)
            ztem_list.append(ztem)
            print('vtem:', vtem)
            vtem_list.append(vtem)

            s = (ztem_list[iter] - Z_list[iter]) / V_list[iter]
            #s = tf.maximum(x = s, y = 0.00001)
            print('s:', s)
            s_list.append(s)

            t = (V_list[iter] - vtem_list[iter]) / tf.square(x = V_list[iter])
            #t = tf.maximum(x = t, y = 0.00001)
            print('t:', t)
            t_list.append(t)

            Sigma = 1. / tf.matmul(a = A_2, b = t_list[iter], transpose_a = True)
            #Sigma = tf.maximum(x = Sigma, y = 0.00001)
            print('Sigma:', Sigma)
            Sigma_list.append(Sigma)

            R = mhat_list[iter] + Sigma_list[iter] * tf.matmul(a = A_val, b = s_list[iter + 1], transpose_a = True)
            #R = tf.maximum(x = R, y = 0.00001)

            print('R:', R)
            R_list.append(R)

            (mhat, vhat, layers) = ResNet_outer_wrapper(R_list[iter], Sigma_list[iter], theta, tie, iter, training=training, LayerbyLayer=LayerbyLayer)
            #vhat = tf.maximum(x = vhat, y = 0.00001)
            print('mhat:', mhat)
            print('vhat:', vhat)

            mhat_list.append(mhat)
            vhat_list.append(vhat)

            layers_list.append(layers)
        #print("------------------- LDGAMP_ResNet-14 -------------------", xhat.shape)
        #print("------------------- LDGAMP_ResNet-15 -------------------", vhat.shape)

        #(xhat,dxdr)=ResNet_outer_wrapper(r, rvar,theta,tie,iter,training=training,LayerbyLayer=LayerbyLayer)
        #if is_complex:
        #    z = y - A_handle(A_val, xhat) + n_fp / m_fp * tf.complex(dxdr,0.) * z
        #else:
        #    z = y - A_handle(A_val, xhat) + n_fp / m_fp * dxdr * z
        #(MSE_thisiter, NMSE_thisiter, PSNR_thisiter) = EvalError(xhat, x_true)
        #MSE_history.append(MSE_thisiter)
        #NMSE_history.append(NMSE_thisiter)
        #PSNR_history.append(PSNR_thisiter)
    return mhat, s_list, mhat_list, vhat_list, V_list, Z_list, ztem_list, vtem_list, t_list, Sigma_list, R_list, layers_list

#Learned DGAMP_ResNet
def LDGAMP_ResNet_2(y,A_handle,At_handle,A_val,theta,x_true, sigma_w, quan_step, n_bit, tie,training=False,LayerbyLayer=True):
    print('------------------------------- LDGAMP_ResNet_2 -------------------------------')
    xhat = tf.zeros_like(x_true, dtype = tf.float32)
    vhat = tf.ones_like(tensor=xhat, dtype=tf.float32)

    xhat_list = []
    xhat_list.append(xhat)

    vhat_list = []
    vhat_list.append(vhat)

    Z = tf.zeros_like(y, dtype = tf.float32)
    V = tf.ones_like(y, dtype = tf.float32)

    V_list = []
    V_list.append(V)

    Z_list = []
    Z_list.append(Z)

    ztem_list = []
    vtem_list = []

    Vtem_list = []
    Ztem_list = []
    Sigma_list = []
    R_list = []

    layers_list = []

    A_2 = A_val * A_val

    add_w = 0.0
    beta = 0.8

    dist = tf.contrib.distributions.Normal(0.0, 1.0)

    for iter in range(n_GAMP_layers):
        print("iter:", iter)

        print('------------------------------- Expection_Variance -------------------------------')
        # ztem, vtem = Expection_Variance(y = y, Z = Z_list[iter], V = V_list[iter], quan_step = quan_step, n_bit = n_bit, sigma_w = sigma_w, dist = dist)
        ztem, vtem, y_up, y_low, eta1, eta2, tmp1, tmp2, tmp3 = \
            Expection_Variance(y=y, Z=Z_list[iter], V=V_list[iter], quan_step=quan_step, n_bit=n_bit, sigma_w=sigma_w,
                               dist=dist)
        # z_hat = tf.maximum(x = z_hat, y = 0.00001)
        # v_hat = tf.maximum(x = v_hat, y = 0.00001)
        print('------------------------------- Expection_Variance -------------------------------')
        print('ztem:', ztem)
        ztem_list.append(ztem)
        print('vtem:', vtem)
        vtem_list.append(vtem)

        Z = Z_list[iter]
        V = V_list[iter]

        xhat = xhat_list[iter]
        vhat = vhat_list[iter]
        for i in range(5):
            Vtem = 1. / (1. / vtem_list[iter] - 1. / V)
            Ztem = Vtem * (ztem_list[iter] / vtem_list[iter] - Z / V)
            Sigma = 1. / tf.matmul(a=A_2, b= 1. / (Vtem + V), transpose_a=True)
            R = xhat + Sigma * tf.matmul(a=A_val, b= (Ztem - Z) / (Vtem + V), transpose_a=True)
            (mhat, vhat, layers) = ResNet_outer_wrapper(R, Sigma, theta, tie, iter,
                training=training, LayerbyLayer=LayerbyLayer)



        V = tf.matmul(a=A_2, b=vhat_list[iter])
        # V = tf.maximum(x = V, y = 0.00001)
        print('V:', V)
        V_list.append(V)

        Z = tf.matmul(a=A_val, b=mhat_list[iter]) - s_list[iter] * V_list[iter]
        # Z = tf.maximum(x = Z, y = 0.00001)

        print('Z:', Z)
        Z_list.append(Z)




        s = (ztem_list[iter] - Z_list[iter]) / V_list[iter]
        # s = tf.maximum(x = s, y = 0.00001)
        print('s:', s)
        s_list.append(s)

        t = (V_list[iter] - vtem_list[iter]) / tf.square(x=V_list[iter])
        # t = tf.maximum(x = t, y = 0.00001)
        print('t:', t)
        t_list.append(t)

        Sigma = 1. / tf.matmul(a=A_2, b=t_list[iter], transpose_a=True)
        # Sigma = tf.maximum(x = Sigma, y = 0.00001)
        print('Sigma:', Sigma)
        Sigma_list.append(Sigma)

        R = mhat_list[iter] + Sigma_list[iter] * tf.matmul(a=A_val, b=s_list[iter + 1], transpose_a=True)
        # R = tf.maximum(x = R, y = 0.00001)

        print('R:', R)
        R_list.append(R)

        (mhat, vhat, layers) = ResNet_outer_wrapper(R_list[iter], Sigma_list[iter], theta, tie, iter, training=training,
                                                   LayerbyLayer=LayerbyLayer)
        # vhat = tf.maximum(x = vhat, y = 0.00001)
        print('mhat:', mhat)
        print('vhat:', vhat)

        mhat_list.append(mhat)
        vhat_list.append(vhat)

        layers_list.append(layers)


    return mhat, s_list, mhat_list, vhat_list, V_list, Z_list, ztem_list, vtem_list, t_list, Sigma_list, R_list, layers_list

def LDGAMP_ResNet_1(y,theta,x_true, sigma_w, quan_step, n_bit, tie,training=False,LayerbyLayer=True):
    print('------------------------------- LDGAMP_ResNet_1 -------------------------------')
    s = tf.zeros(shape = (m, BATCH_SIZE), dtype = tf.float32)
    #s = y
    print('s:', s)

    mhat_old = tf.reduce_mean(input_tensor = x_true, axis = 1, keepdims = True)
    print('mhat_old:', mhat_old)
    mhat = tf.tile(input = mhat_old, multiples = [1, BATCH_SIZE])
    print('mhat:', mhat)

    print('BATCH_SIZE:', BATCH_SIZE)

    if 1 == BATCH_SIZE:
        print('!!!!!!!!!!!!!!!!!!!!!!')
        vhat = tf.ones_like(tensor = mhat, dtype = tf.float32)
        print('vhat:', vhat)
    else:
        vhat_old = tf.reduce_mean(input_tensor = tf.square(x = x_true - mhat_old), axis = 1, keepdims = True)
        print('vhat_old:', vhat_old)
        vhat = tf.tile(input = vhat_old, multiples = [1, BATCH_SIZE])
        print('vhat:', vhat)

    s_list = []
    s_list.append(s)

    mhat_list = []
    mhat_list.append(mhat)

    vhat_list = []
    vhat_list.append(vhat)

    V_list = []
    Z_list = []
    ztem_list = []
    vtem_list = []

    t_list = []
    Sigma_list = []
    R_list = []

    layers_list = []

    #A_2 = A_val * A_val

    add_w = 0.0
    beta = 0.8

    dist = tf.contrib.distributions.Normal(0.0, 1.0)

    for iter in range(n_GAMP_layers):
        print("iter:", iter)
        if is_complex:
            r = tf.complex(xhat,tf.zeros([n,BATCH_SIZE],dtype=tf.float32)) + At_handle(A_val,z)
            rvar = (1. / m_fp * tf.reduce_sum(tf.square(tf.abs(z)),axis=0))#In the latest version of TF, abs can handle complex values
        else:
            #V = tf.matmul(a = A_2, b = vhat_list[iter])
            V = vhat_list[iter]
            #V = tf.maximum(x = V, y = 0.00001)
            print('V:', V)
            V_list.append(V)

            #Z = tf.matmul(a = A_val, b = mhat_list[iter]) - s_list[iter] * V_list[iter]
            Z = mhat_list[iter] - s_list[iter] * V_list[iter]
            #Z = tf.maximum(x = Z, y = 0.00001)

            print('Z:', Z)
            Z_list.append(Z)
            print('------------------------------- Expection_Variance -------------------------------')
            #ztem, vtem = Expection_Variance(y = y, Z = Z_list[iter], V = V_list[iter], quan_step = quan_step, n_bit = n_bit, sigma_w = sigma_w, dist = dist)
            ztem, vtem, y_up, y_low, eta1, eta2, tmp1, tmp2, tmp3 = \
                Expection_Variance(y=y, Z=Z_list[iter], V=V_list[iter], quan_step=quan_step, n_bit=n_bit, sigma_w=sigma_w, dist=dist)
            #z_hat = tf.maximum(x = z_hat, y = 0.00001)
            #v_hat = tf.maximum(x = v_hat, y = 0.00001)

            print('------------------------------- Expection_Variance -------------------------------')

            print('ztem:', ztem)
            ztem_list.append(ztem)
            print('vtem:', vtem)
            vtem_list.append(vtem)

            s = (ztem_list[iter] - Z_list[iter]) / V_list[iter]
            #s = tf.maximum(x = s, y = 0.00001)
            print('s:', s)
            s_list.append(s)

            t = (V_list[iter] - vtem_list[iter]) / tf.square(x = V_list[iter])
            #t = tf.maximum(x = t, y = 0.00001)
            print('t:', t)
            t_list.append(t)

            #Sigma = 1. / tf.matmul(a = A_2, b = t_list[iter], transpose_a = True)
            Sigma = 1. / t_list[iter]
            #Sigma = tf.maximum(x = Sigma, y = 0.00001)
            print('Sigma:', Sigma)
            Sigma_list.append(Sigma)

            #R = mhat_list[iter] + Sigma_list[iter] * tf.matmul(a = A_val, b = s_list[iter + 1], transpose_a = True)
            R = mhat_list[iter] + Sigma_list[iter] * s_list[iter + 1]
            #R = tf.maximum(x = R, y = 0.00001)

            print('R:', R)
            R_list.append(R)

            (mhat, vhat, layers) = ResNet_outer_wrapper(R_list[iter], Sigma_list[iter], theta, tie, iter, training=training, LayerbyLayer=LayerbyLayer)
            #vhat = tf.maximum(x = vhat, y = 0.00001)
            print('mhat:', mhat)
            print('vhat:', vhat)

            mhat_list.append(mhat)
            vhat_list.append(vhat)

            layers_list.append(layers)
        #print("------------------- LDGAMP_ResNet-14 -------------------", xhat.shape)
        #print("------------------- LDGAMP_ResNet-15 -------------------", vhat.shape)

        #(xhat,dxdr)=ResNet_outer_wrapper(r, rvar,theta,tie,iter,training=training,LayerbyLayer=LayerbyLayer)
        #if is_complex:
        #    z = y - A_handle(A_val, xhat) + n_fp / m_fp * tf.complex(dxdr,0.) * z
        #else:
        #    z = y - A_handle(A_val, xhat) + n_fp / m_fp * dxdr * z
        #(MSE_thisiter, NMSE_thisiter, PSNR_thisiter) = EvalError(xhat, x_true)
        #MSE_history.append(MSE_thisiter)
        #NMSE_history.append(NMSE_thisiter)
        #PSNR_history.append(PSNR_thisiter)
    return mhat, s_list, mhat_list, vhat_list, V_list, Z_list, ztem_list, vtem_list, t_list, Sigma_list, R_list, layers_list

#Learned DGAMP_ResNet operating on Aty. Used for calculating MCSURE loss
def LDGAMP_ResNet_Aty(Aty,A_handle,At_handle,A_val,theta,x_true,tie,training=False,LayerbyLayer=True):
    Atz=Aty
    xhat = tf.zeros([n, BATCH_SIZE], dtype=tf.float32)
    MSE_history=[]#Will be a list of n_GAMP_layers+1 lists, each sublist will be of size BATCH_SIZE
    NMSE_history=[]
    PSNR_history=[]
    (MSE_thisiter, NMSE_thisiter, PSNR_thisiter)=EvalError(xhat,x_true)
    MSE_history.append(MSE_thisiter)
    NMSE_history.append(NMSE_thisiter)
    PSNR_history.append(PSNR_thisiter)
    for iter in range(n_GAMP_layers):
        if is_complex:
            r = tf.complex(xhat, tf.zeros([n, BATCH_SIZE], dtype=tf.float32)) + Atz
            rvar = (1. / n_fp * tf.reduce_sum(tf.square(tf.abs(Atz)), axis=0))
            # rvar = (1. / m_fp * tf.reduce_sum(tf.square(tf.abs(z)),axis=0))#In the latest version of TF, abs can handle complex values
        else:
            r = xhat + Atz
            rvar = (1. / n_fp * tf.reduce_sum(tf.square(tf.abs(Atz)), axis=0))
            # rvar = (1. / m_fp * tf.reduce_sum(tf.square(tf.abs(z)),axis=0))
        (xhat,dxdr)=ResNet_outer_wrapper(r, rvar,theta,tie,iter,training=training,LayerbyLayer=LayerbyLayer)
        if is_complex:
            # z = y - A_handle(A_val, xhat) + n_fp / m_fp * tf.complex(dxdr,0.) * z
            Atz = Aty - At_handle(A_val, A_handle(A_val, xhat)) + n_fp / m_fp * tf.complex(dxdr,0.) * Atz
        else:
            # z = y - A_handle(A_val, xhat) + n_fp / m_fp * dxdr * z
            Atz = Aty - At_handle(A_val, A_handle(A_val, xhat)) +  n_fp / m_fp * dxdr * Atz
        (MSE_thisiter, NMSE_thisiter, PSNR_thisiter) = EvalError(xhat, x_true)
        MSE_history.append(MSE_thisiter)
        NMSE_history.append(NMSE_thisiter)
        PSNR_history.append(PSNR_thisiter)
    return xhat, MSE_history, NMSE_history, PSNR_history, r, rvar, dxdr

#Learned DIT
def LDIT(y,A_handle,At_handle,A_val,theta,x_true,tie,training=False,LayerbyLayer=True):
    z=y
    xhat = tf.zeros([n, BATCH_SIZE], dtype=tf.float32)
    MSE_history=[]#Will be a list of n_GAMP_layers+1 lists, each sublist will be of size BATCH_SIZE
    NMSE_history=[]
    PSNR_history=[]
    (MSE_thisiter, NMSE_thisiter, PSNR_thisiter)=EvalError(xhat,x_true)
    MSE_history.append(MSE_thisiter)
    NMSE_history.append(NMSE_thisiter)
    PSNR_history.append(PSNR_thisiter)
    for iter in range(n_GAMP_layers):
        if is_complex:
            r = tf.complex(xhat,tf.zeros([n,BATCH_SIZE],dtype=tf.float32)) + At_handle(A_val,z)
            rvar = (1. / m_fp * tf.reduce_sum(tf.square(tf.abs(z)),axis=0))#In the latest version of TF, abs can handle complex values
        else:
            r = xhat + At_handle(A_val,z)
            rvar = (1. / m_fp * tf.reduce_sum(tf.square(tf.abs(z)),axis=0))
        (xhat, dxdr) = ResNet_outer_wrapper(r, 4.*rvar, theta, tie, iter,training=training,LayerbyLayer=LayerbyLayer)
        if is_complex:
            z = y - A_handle(A_val, xhat)
        else:
            z = y - A_handle(A_val, xhat)
        (MSE_thisiter, NMSE_thisiter, PSNR_thisiter) = EvalError(xhat, x_true)
        MSE_history.append(MSE_thisiter)
        NMSE_history.append(NMSE_thisiter)
        PSNR_history.append(PSNR_thisiter)
    return xhat, MSE_history, NMSE_history, PSNR_history

#Learned DGAMP
def LDGAMPxasxasxxas(y,A_handle,At_handle,A_val,theta,x_true,tie,training=False,LayerbyLayer=True):
    # GAMP notation is used here
    # LDGAMP does not presently support function handles: It does not work with the latest version of the code.
    wvar = tf.square(sigma_w)#Assume noise level is known. Could be learned
    Beta = 1.0 # For now perform no GAMPing
    xhat = tf.zeros([n,BATCH_SIZE], dtype=tf.float32)
    xbar = xhat
    xvar = tf.ones((1,BATCH_SIZE), dtype=tf.float32)
    s = .00001*tf.ones((m, BATCH_SIZE), dtype=tf.float32)
    svar = .00001 * tf.ones((1, BATCH_SIZE), dtype=tf.float32)
    pvar = .00001 * tf.ones((1, BATCH_SIZE), dtype=tf.float32)
    OneOverM = tf.constant(float(1) / m, dtype=tf.float32)
    A_norm2=tf.reduce_sum(tf.square(A_val))
    OneOverA_norm2 = 1. / A_norm2
    MSE_history=[]#Will be a list of n_GAMP_layers+1 lists, each sublist will be of size BATCH_SIZE
    NMSE_history=[]
    PSNR_history=[]
    (MSE_thisiter, NMSE_thisiter, PSNR_thisiter)=EvalError(xhat,x_true)
    MSE_history.append(MSE_thisiter)
    NMSE_history.append(NMSE_thisiter)
    PSNR_history.append(PSNR_thisiter)
    for iter in range(n_GAMP_layers):
        pvar = Beta * A_norm2 * OneOverM * xvar + (1 - Beta) * pvar
        pvar = tf.maximum(pvar, .00001)
        p = tf.matmul(A_val, xhat) - pvar * s
        g, dg = g_out_gaussian(p, pvar, y, wvar)
        s = Beta * g + (1 - Beta) * s
        svar = -Beta * dg + (1 - Beta) * svar
        svar = tf.maximum(svar, .00001)
        rvar = OneOverA_norm2 * n / svar
        rvar = tf.maximum(rvar, .00001)
        xbar = Beta * xhat + (1 - Beta) * xbar
        r = xbar + rvar * tf.matmul(A_val, s,adjoint_a=True)
        (xhat, dxdr) = ResNet_outer_wrapper(r, rvar, theta, tie, iter,training=training,LayerbyLayer=LayerbyLayer)
        xvar = dxdr * rvar
        xvar = tf.maximum(xvar, .00001)
        (MSE_thisiter, NMSE_thisiter, PSNR_thisiter) = EvalError(xhat, x_true)
        MSE_history.append(MSE_thisiter)
        NMSE_history.append(NMSE_thisiter)
        PSNR_history.append(PSNR_thisiter)
    return xhat, MSE_history, NMSE_history, PSNR_history

def init_vars_ResNet(init_mu,init_sigma):

    weights = [None] * 26
    biases = [None] * 26

    i = 0
    with tf.variable_scope("conv_" + str(i)):
        weights[0] = tf.Variable(
            tf.truncated_normal(shape=(3, 3, 1, 16), mean=init_mu,
            stddev=init_sigma, dtype=tf.float32), dtype=tf.float32, name="w")
        biases[0] = tf.Variable(tf.zeros(16), dtype=tf.float32, name="b")

    num = 3
    i = 1
    with tf.variable_scope("conv_" + str(i)):
        weights[num * (i - 1) + 1] = tf.Variable(
            tf.truncated_normal(shape=(1, 1, 16, 16), mean=init_mu,
            stddev=init_sigma, dtype=tf.float32), dtype=tf.float32, name="w0")
        biases[num * (i - 1) + 1] = tf.Variable(tf.zeros(16), dtype=tf.float32, name="b0")
        weights[num * (i - 1) + 2] = tf.Variable(
            tf.truncated_normal(shape=(3, 3, 16, 16), mean=init_mu,
            stddev=init_sigma, dtype=tf.float32), dtype=tf.float32, name="w1")
        biases[num * (i - 1) + 2] = tf.Variable(tf.zeros(16), dtype=tf.float32, name="b1")
        weights[num * (i - 1) + 3] = tf.Variable(
            tf.truncated_normal(shape=(3, 3, 16, 16), mean=init_mu,
            stddev=init_sigma, dtype=tf.float32), dtype=tf.float32, name="w2")
        biases[num * (i - 1) + 3] = tf.Variable(tf.zeros(16), dtype=tf.float32, name="b2")

    i = 2
    with tf.variable_scope("conv_" + str(i)):
        weights[num * (i - 1) + 1] = tf.Variable(
            tf.truncated_normal(shape=(1, 1, 16, 16), mean=init_mu,
            stddev=init_sigma, dtype=tf.float32), dtype=tf.float32, name="w0")
        biases[num * (i - 1) + 1] = tf.Variable(tf.zeros(16), dtype=tf.float32, name="b0")
        weights[num * (i - 1) + 2] = tf.Variable(
            tf.truncated_normal(shape=(3, 3, 16, 16), mean=init_mu,
            stddev=init_sigma, dtype=tf.float32), dtype=tf.float32, name="w1")
        biases[num * (i - 1) + 2] = tf.Variable(tf.zeros(16), dtype=tf.float32, name="b1")
        weights[num * (i - 1) + 3] = tf.Variable(
            tf.truncated_normal(shape=(3, 3, 16, 16), mean=init_mu,
            stddev=init_sigma, dtype=tf.float32), dtype=tf.float32, name="w2")
        biases[num * (i - 1) + 3] = tf.Variable(tf.zeros(16), dtype=tf.float32, name="b2")

    i = 3
    with tf.variable_scope("conv_" + str(i)):
        weights[num * (i - 1) + 1] = tf.Variable(
            tf.truncated_normal(shape=(1, 1, 16, 32), mean=init_mu,
            stddev=init_sigma, dtype=tf.float32), dtype=tf.float32, name="w0")
        biases[num * (i - 1) + 1] = tf.Variable(tf.zeros(32), dtype=tf.float32, name="b0")
        weights[num * (i - 1) + 2] = tf.Variable(
            tf.truncated_normal(shape=(3, 3, 16, 32), mean=init_mu,
            stddev=init_sigma, dtype=tf.float32), dtype=tf.float32, name="w1")
        biases[num * (i - 1) + 2] = tf.Variable(tf.zeros(32), dtype=tf.float32, name="b1")
        weights[num * (i - 1) + 3] = tf.Variable(
            tf.truncated_normal(shape=(3, 3, 32, 32), mean=init_mu,
            stddev=init_sigma, dtype=tf.float32), dtype=tf.float32, name="w2")
        biases[num * (i - 1) + 3] = tf.Variable(tf.zeros(32), dtype=tf.float32, name="b2")

    i = 4
    with tf.variable_scope("conv_" + str(i)):
        weights[num * (i - 1) + 1] = tf.Variable(
            tf.truncated_normal(shape=(1, 1, 32, 32), mean=init_mu,
            stddev=init_sigma, dtype=tf.float32), dtype=tf.float32, name="w0")
        biases[num * (i - 1) + 1] = tf.Variable(tf.zeros(32), dtype=tf.float32, name="b0")
        weights[num * (i - 1) + 2] = tf.Variable(
            tf.truncated_normal(shape=(3, 3, 32, 32), mean=init_mu,
            stddev=init_sigma, dtype=tf.float32), dtype=tf.float32, name="w1")
        biases[num * (i - 1) + 2] = tf.Variable(tf.zeros(32), dtype=tf.float32, name="b1")
        weights[num * (i - 1) + 3] = tf.Variable(
            tf.truncated_normal(shape=(3, 3, 32, 32), mean=init_mu,
            stddev=init_sigma, dtype=tf.float32), dtype=tf.float32, name="w2")
        biases[num * (i - 1) + 3] = tf.Variable(tf.zeros(32), dtype=tf.float32, name="b2")

    i = 5
    with tf.variable_scope("conv_" + str(i)):
        weights[num * (i - 1) + 1] = tf.Variable(
            tf.truncated_normal(shape=(1, 1, 32, 48), mean=init_mu,
            stddev=init_sigma, dtype=tf.float32), dtype=tf.float32, name="w0")
        biases[num * (i - 1) + 1] = tf.Variable(tf.zeros(48), dtype=tf.float32, name="b0")
        weights[num * (i - 1) + 2] = tf.Variable(
            tf.truncated_normal(shape=(3, 3, 32, 48), mean=init_mu,
            stddev=init_sigma, dtype=tf.float32), dtype=tf.float32, name="w1")
        biases[num * (i - 1) + 2] = tf.Variable(tf.zeros(48), dtype=tf.float32, name="b1")
        weights[num * (i - 1) + 3] = tf.Variable(
            tf.truncated_normal(shape=(3, 3, 48, 48), mean=init_mu,
            stddev=init_sigma, dtype=tf.float32), dtype=tf.float32, name="w2")
        biases[num * (i - 1) + 3] = tf.Variable(tf.zeros(48), dtype=tf.float32, name="b2")

    i = 6
    with tf.variable_scope("conv_" + str(i)):
        weights[num * (i - 1) + 1] = tf.Variable(
            tf.truncated_normal(shape=(1, 1, 48, 48), mean=init_mu,
            stddev=init_sigma, dtype=tf.float32), dtype=tf.float32, name="w0")
        biases[num * (i - 1) + 1] = tf.Variable(tf.zeros(48), dtype=tf.float32, name="b0")
        weights[num * (i - 1) + 2] = tf.Variable(
            tf.truncated_normal(shape=(3, 3, 48, 48), mean=init_mu,
            stddev=init_sigma, dtype=tf.float32), dtype=tf.float32, name="w1")
        biases[num * (i - 1) + 2] = tf.Variable(tf.zeros(48), dtype=tf.float32, name="b1")
        weights[num * (i - 1) + 3] = tf.Variable(
            tf.truncated_normal(shape=(3, 3, 48, 48), mean=init_mu,
            stddev=init_sigma, dtype=tf.float32), dtype=tf.float32, name="w2")
        biases[num * (i - 1) + 3] = tf.Variable(tf.zeros(48), dtype=tf.float32, name="b2")

    i = 7
    with tf.variable_scope("conv_" + str(i)):
        weights[num * (i - 1) + 1] = tf.Variable(
            tf.truncated_normal(shape=(1, 1, 48, 64), mean=init_mu,
            stddev=init_sigma, dtype=tf.float32), dtype=tf.float32, name="w0")
        biases[num * (i - 1) + 1] = tf.Variable(tf.zeros(64), dtype=tf.float32, name="b0")
        weights[num * (i - 1) + 2] = tf.Variable(
            tf.truncated_normal(shape=(3, 3, 48, 64), mean=init_mu,
            stddev=init_sigma, dtype=tf.float32), dtype=tf.float32, name="w1")
        biases[num * (i - 1) + 2] = tf.Variable(tf.zeros(64), dtype=tf.float32, name="b1")
        weights[num * (i - 1) + 3] = tf.Variable(
            tf.truncated_normal(shape=(3, 3, 64, 64), mean=init_mu,
            stddev=init_sigma, dtype=tf.float32), dtype=tf.float32, name="w2")
        biases[num * (i - 1) + 3] = tf.Variable(tf.zeros(64), dtype=tf.float32, name="b2")

    i = 8
    with tf.variable_scope("conv_" + str(i)):
        weights[num * (i - 1) + 1] = tf.Variable(
            tf.truncated_normal(shape=(1, 1, 64, 64), mean=init_mu,
            stddev=init_sigma, dtype=tf.float32), dtype=tf.float32, name="w0")
        biases[num * (i - 1) + 1] = tf.Variable(tf.zeros(64), dtype=tf.float32, name="b0")
        weights[num * (i - 1) + 2] = tf.Variable(
            tf.truncated_normal(shape=(3, 3, 64, 64), mean=init_mu,
            stddev=init_sigma, dtype=tf.float32), dtype=tf.float32, name="w1")
        biases[num * (i - 1) + 2] = tf.Variable(tf.zeros(64), dtype=tf.float32, name="b1")
        weights[num * (i - 1) + 3] = tf.Variable(
            tf.truncated_normal(shape=(3, 3, 64, 64), mean=init_mu,
            stddev=init_sigma, dtype=tf.float32), dtype=tf.float32, name="w2")
        biases[num * (i - 1) + 3] = tf.Variable(tf.zeros(64), dtype=tf.float32, name="b2")

    i = 9
    with tf.variable_scope("conv_" + str(i)):
        weights[-1] = tf.Variable(
            tf.truncated_normal(shape=(1, 1, 64, 1), mean=init_mu,
            stddev=init_sigma, dtype=tf.float32), dtype=tf.float32, name="w")
        biases[-1] = tf.Variable(tf.zeros(1), dtype=tf.float32, name="b")

    return weights, biases #, betas, moving_variances, moving_means

## Evaluate Intermediate Error
def EvalError(x_hat,x_true):
    mse=tf.reduce_mean(tf.square(x_hat-x_true),axis=0)
    xnorm2=tf.reduce_mean(tf.square( x_true),axis=0)
    mse_thisiter=mse
    nmse_thisiter=mse/xnorm2
    psnr_thisiter=10.*tf.log(1./mse)/tf.log(10.)
    return mse_thisiter, nmse_thisiter, psnr_thisiter

## Evaluate Intermediate Error
def EvalError_np(x_hat,x_true):
    mse=np.mean(np.square(x_hat-x_true),axis=0)
    xnorm2=np.mean(np.square( x_true),axis=0)
    mse_thisiter=mse
    nmse_thisiter=mse/xnorm2
    psnr_thisiter=10.*np.log(1./mse)/np.log(10.)
    return mse_thisiter, nmse_thisiter, psnr_thisiter

## Output Denoiser Gaussian
def g_out_gaussian(phat,pvar,y,wvar):
    g=(y-phat)*1/(pvar+wvar)
    dg=-1/(pvar+wvar)
    return g, dg

## Output Denoiser Rician
def g_out_phaseless(phat,pvar,y,wvar):
    #Untested. To be used with D-prGAMP

    y_abs = y
    phat_abs = tf.abs(phat)
    B = 2.* tf.div(tf.multiply(y_abs,phat_abs),wvar+pvar)
    I1overI0 = tf.minimum(tf.div(B,tf.sqrt(tf.square(B)+4)),tf.div(B,0.5+tf.sqrt(tf.square(B)+0.25)))
    y_sca = tf.div(y_abs,1.+tf.div(wvar,pvar))
    phat_sca = tf.div(phat_abs,1.+tf.div(pvar,wvar))
    zhat = tf.multiply(tf.add(phat_sca,tf.multiply(y_sca,I1overI0)),tf.sign(phat))

    sigma2_z = tf.add(tf.add(tf.square(y_sca),tf.square(phat_sca)),tf.subtract(tf.div(1.+tf.multiply(B,I1overI0),tf.add(tf.div(1.,wvar),tf.div(1.,pvar))),tf.square(tf.abs(zhat))))

    g = tf.multiply(tf.div(1.,pvar),tf.subtract(zhat,phat))
    dg = tf.multiply(tf.div(1.,pvar),tf.subtract(tf.div(tf.reduce_mean(sigma2_z,axis=(1,2,3)),pvar),1))

    return g,dg

## Denoiser wrapper that selects which weights and biases to use
def ResNet_outer_wrapper(R, Sigma, theta, tie, iter, training=False, LayerbyLayer=True):
#def ResNet_outer_wrapper(r,rvar,theta,tie,iter,training=False,LayerbyLayer=True):
    if tie:
        with tf.variable_scope("Iter0"):
            (xhat, dxdr) = ResNet_wrapper(r, rvar, theta[0], training=training)
    elif adaptive_weights:
        rstd = 255. * tf.sqrt(tf.reduce_mean(rvar))  # To enable batch processing, I have to treat every image in the batch as if it has the same amount of effective noise
        def x_nl0(a=rstd,iter=iter,r=r,rvar=rvar,theta=theta):
            with tf.variable_scope("Adaptive_NL" + str(0)):
                (xhat, dxdr) = ResNet_wrapper(r, rvar, theta[0], training=training)
            xhat = tf.Print(xhat, [iter], "used denoiser 0")
            return (xhat, dxdr)

        def x_nl1(a=rstd,iter=iter,r=r,rvar=rvar,theta=theta):
            with tf.variable_scope("Adaptive_NL" + str(1)):
                (xhat, dxdr) = ResNet_wrapper(r, rvar, theta[1], training=training)
            xhat = tf.Print(xhat, [iter], "used denoiser 1")
            return (xhat, dxdr)

        def x_nl2(a=rstd,iter=iter,r=r,rvar=rvar,theta=theta):
            with tf.variable_scope("Adaptive_NL" + str(2)):
                (xhat, dxdr) = ResNet_wrapper(r, rvar, theta[2], training=training)
            xhat = tf.Print(xhat, [iter], "used denoiser 2")
            return (xhat, dxdr)

        def x_nl3(a=rstd,iter=iter,r=r,rvar=rvar,theta=theta):
            with tf.variable_scope("Adaptive_NL" + str(3)):
                (xhat, dxdr) = ResNet_wrapper(r, rvar, theta[3], training=training)
            xhat = tf.Print(xhat, [iter], "used denoiser 3")
            return (xhat, dxdr)

        def x_nl4(a=rstd,iter=iter,r=r,rvar=rvar,theta=theta):
            with tf.variable_scope("Adaptive_NL" + str(4)):
                (xhat, dxdr) = ResNet_wrapper(r, rvar, theta[4], training=training)
            xhat = tf.Print(xhat, [iter], "used denoiser 4")
            return (xhat, dxdr)

        def x_nl5(a=rstd,iter=iter,r=r,rvar=rvar,theta=theta):
            with tf.variable_scope("Adaptive_NL" + str(5)):
                (xhat, dxdr) = ResNet_wrapper(r, rvar, theta[5], training=training)
            xhat = tf.Print(xhat, [iter], "used denoiser 5")
            return (xhat, dxdr)

        def x_nl6(a=rstd,iter=iter,r=r,rvar=rvar,theta=theta):
            with tf.variable_scope("Adaptive_NL" + str(6)):
                (xhat, dxdr) = ResNet_wrapper(r, rvar, theta[6], training=training)
            xhat = tf.Print(xhat, [iter], "used denoiser 6")
            return (xhat, dxdr)

        def x_nl7(a=rstd,iter=iter,r=r,rvar=rvar,theta=theta):
            with tf.variable_scope("Adaptive_NL" + str(7)) as scope:
                (xhat, dxdr) = ResNet_wrapper(r, rvar, theta[7], training=training)
            xhat = tf.Print(xhat, [iter], "used denoiser 7")
            return (xhat, dxdr)

        def x_nl8(a=rstd,iter=iter,r=r,rvar=rvar,theta=theta):
            with tf.variable_scope("Adaptive_NL" + str(8)) as scope:
                (xhat, dxdr) = ResNet_wrapper(r, rvar, theta[8], training=training)
            xhat = tf.Print(xhat, [iter], "used denoiser 8")
            return (xhat, dxdr)

        rstd=tf.Print(rstd,[rstd],"rstd =")
        NL_0 = tf.less_equal(rstd, 10.)
        NL_1 = tf.logical_and(tf.less(10., rstd), tf.less_equal(rstd, 20.))
        NL_2 = tf.logical_and(tf.less(20., rstd), tf.less_equal(rstd, 40.))
        NL_3 = tf.logical_and(tf.less(40., rstd), tf.less_equal(rstd, 60.))
        NL_4 = tf.logical_and(tf.less(60., rstd), tf.less_equal(rstd, 80.))
        NL_5 = tf.logical_and(tf.less(80., rstd), tf.less_equal(rstd, 100.))
        NL_6 = tf.logical_and(tf.less(100., rstd), tf.less_equal(rstd, 150.))
        NL_7 = tf.logical_and(tf.less(150., rstd), tf.less_equal(rstd, 300.))
        predicates = {NL_0:  x_nl0, NL_1:  x_nl1, NL_2:  x_nl2, NL_3:  x_nl3, NL_4:  x_nl4, NL_5:  x_nl5, NL_6:  x_nl6, NL_7:  x_nl7}
        default =  x_nl8
        (xhat,dxdr) = tf.case(predicates,default,exclusive=True)
        xhat = tf.reshape(xhat, shape=[n, BATCH_SIZE])
        dxdr = tf.reshape(dxdr, shape=[1, BATCH_SIZE])
    else:
        with tf.variable_scope("Iter" + str(iter)):
            (xhat, dxdr, layers) = ResNet_wrapper(R, Sigma, theta[iter], training=training, LayerbyLayer=LayerbyLayer)
            #(xhat, dxdr) = ResNet_wrapper(r, rvar, theta[iter], training=training,LayerbyLayer=LayerbyLayer)
    return (xhat, dxdr, layers)

## Denoiser Wrapper that computes divergence
def ResNet_wrapper(R, Sigma,theta_thislayer,training=False,LayerbyLayer=True):
#def ResNet_wrapper(r,rvar,theta_thislayer,training=False,LayerbyLayer=True):
    """
    Call a black-box denoiser and compute a Monte Carlo estimate of dx/dr
    """
    (xhat, layers) = ResNet(R, Sigma,theta_thislayer,training=training)
    r_abs = tf.abs(R, name=None)
    #epsilon = tf.maximum(.001 * tf.reduce_max(r_abs, axis=0),.00001)
    epsilon = tf.maximum(.001 * tf.reduce_max(r_abs, axis = 0, keep_dims = True), .00001)

    eta=tf.random_normal(shape=R.get_shape(),dtype=tf.float32)
    if is_complex:
        r_perturbed = r + tf.complex(tf.multiply(eta, epsilon),tf.zeros([n,BATCH_SIZE],dtype=tf.float32))
    else:
        #r_perturbed = r + tf.multiply(eta, epsilon)
        r_perturbed = R + tf.multiply(eta, epsilon)

    (xhat_perturbed, _) = ResNet(r_perturbed, Sigma, theta_thislayer,training=training)  # Avoid computing gradients wrt this use of theta_thislayer

    eta_dx = tf.multiply(eta, xhat_perturbed - xhat)
    #mean_eta_dx = tf.reduce_mean(eta_dx, axis=0)
    mean_eta_dx = tf.reduce_mean(eta_dx, axis = 0, keepdims = True)
    dxdrMC = tf.divide(mean_eta_dx, epsilon)
    dxdrMC = tf.tile(input = dxdrMC, multiples = [R.get_shape()[0], 1])
    dxdrMC = dxdrMC * Sigma
    dxdrMC = tf.maximum(dxdrMC, .00001)

    if not LayerbyLayer:
        dxdrMC=tf.stop_gradient(dxdrMC)#When training long networks end-to-end propagating wrt the MC estimates caused divergence
    return(xhat, dxdrMC, layers)

## Create Denoiser Model
def ResNet(R, Sigma, theta_thislayer,training=False):
    # def ResNet(r,rvar, theta_thislayer,training=False):
    # Reuse will always be true, thus init_vars_ResNet must be called within the appropriate namescope before ResNet can be used
    ##r is n x batch_size, where in this case n would be height_img*width_img*channel_img
    # rvar is unused within ResNet. It may have been used to select which sets of weights and biases to use.
    weights = theta_thislayer[0]
    biases = theta_thislayer[1]

    if is_complex:
        r = tf.real(r)

    r = tf.transpose(R)
    orig_Shape = tf.shape(r)
    shape4D = [-1, height_img, width_img, channel_img]
    r = tf.reshape(r, shape4D)
    x = r

    num = 9

    i = 0
    with tf.variable_scope("conv_" + str(i)):
        x = tf.nn.conv2d(x, weights[0], strides=[1, 1, 1, 1], padding='SAME')
        x = tf.nn.relu(x)

    for i in range(1, num):
        with tf.variable_scope("conv_" + str(i)):
            p = x

            x = tf.nn.conv2d(p, weights[3 * (i - 1) + 1], strides=[1, 1, 1, 1], padding='SAME')
            x = tf.layers.batch_normalization(inputs=x, training=training, name='BN0', reuse=tf.AUTO_REUSE)
            x = tf.nn.relu(x)

            shortcut = x

            x = tf.nn.conv2d(p, weights[3 * (i - 1) + 2], strides=[1, 1, 1, 1], padding='SAME')
            x = tf.layers.batch_normalization(inputs=x, training=training, name='BN1', reuse=tf.AUTO_REUSE)
            x = tf.nn.relu(x)

            x = tf.nn.conv2d(x, weights[3 * (i - 1) + 3], strides=[1, 1, 1, 1], padding='SAME')
            x = tf.layers.batch_normalization(inputs=x, training=training, name='BN2', reuse=tf.AUTO_REUSE)
            x = tf.nn.relu(x)

            x += shortcut

    i = num
    with tf.variable_scope("conv_" + str(i)):
        x = tf.nn.conv2d(x, weights[-1], strides=[1, 1, 1, 1], padding='SAME')
    x_hat = r - x

    print("x_hat:", x_hat)

    x_hat = tf.transpose(tf.reshape(x_hat, orig_Shape))
    return x_hat, None

# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!! #
## Create training data from images, with tf and function handles
def GenerateNoisyCSData_handles(x,A_handle,sigma_w,A_params, n_bit):
    print('------------------------------- GenerateNoisyCSData_handles -------------------------------')

    z = A_handle(A_params,x)
    z_w, noise_vec = AddNoise(z,sigma_w)
    print('z_w:', z_w)

    # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!! #
    if 1.0 == n_bit:
        quan_step = 2.0
    elif 2.0 == n_bit:
        quan_step = 0.5
    else:
        #quan_step = (tf.reduce_max(input_tensor = tf.abs(x = y_1), axis = 0, keepdims = True) + 1.0) / math.pow(2, n_bit)
        quan_step = 2.0 / np.power(2.0, n_bit)
    print('quan_step:', quan_step)

    tmp1 = np.arange(start = 0.0, stop = np.power(2.0, n_bit - 1.0), step = 1.0, dtype = np.float32)
    tmp1 = np.reshape(a = tmp1, newshape = (tmp1.shape[0], 1))
    print('tmp1:', tmp1)

    tmp2 = np.arange(start = 1.0, stop = np.power(2.0, n_bit), step = 2.0, dtype = np.float32)
    tmp2 = np.reshape(a = tmp2, newshape = (tmp2.shape[0], 1))
    print('tmp2:', tmp2)

    if 1.0 == n_bit or 2.0 == n_bit:
        DeltaTh = tmp1 * quan_step
        Q_out = tmp2 * quan_step / 2.0
    else:
        #DeltaTh = tf.matmul(a = tmp1, b = quan_step)
        #Q_out = tf.matmul(a = tmp2, b = quan_step / 2.0)
        DeltaTh = tmp1 * quan_step
        Q_out = tmp2 * quan_step / 2.0
    print('DeltaTh:', DeltaTh)
    print('Q_out:', Q_out)

    if 1.0 == n_bit or 2.0 == n_bit:
        print('DeltaTh[-1, 0]:', DeltaTh[-1, 0])
        print('Q_out[-1, 0]:', Q_out[-1, 0])
        y_R = Q_out[-1, 0] * tf.cast(x = tf.greater_equal(x = tf.abs(x = z_w), y = DeltaTh[-1, 0]), dtype = tf.float32)
        for bIdx in reversed(range(Q_out.shape[0] - 1)):
            print('bIdx:', bIdx)
            y_R += Q_out[bIdx, 0] * tf.cast(x = tf.greater_equal(x = tf.abs(x = z_w), y = DeltaTh[bIdx, 0]), dtype = tf.float32) * \
                tf.cast(x = tf.less(x = tf.abs(x = z_w), y = DeltaTh[bIdx + 1, 0]), dtype = tf.float32)
    else:
        print('DeltaTh[-1]:', DeltaTh[-1])
        print('Q_out[-1]:', Q_out[-1])
        y_R = Q_out[-1] * tf.cast(x = tf.greater_equal(x = tf.abs(x = z_w), y = DeltaTh[-1]), dtype = tf.float32)
        for bIdx in range(DeltaTh.shape[0] - 1):
            #print('bIdx:', bIdx)
            y_R += Q_out[bIdx] * tf.cast(x = tf.greater_equal(x = tf.abs(x = z_w), y = DeltaTh[bIdx]), dtype = tf.float32) * \
                tf.cast(x = tf.less(x = tf.abs(x = z_w), y = DeltaTh[bIdx + 1]), dtype = tf.float32)

    print('y_R:', y_R)
    y = tf.sign(x = z_w) * y_R
    print('y:', y)
    return z, z_w, noise_vec, quan_step, DeltaTh, Q_out, y_R, y

def GenerateNoisyCSData_handles_1(x, sigma_w, n_bit):
    print('------------------------------- GenerateNoisyCSData_handles_1 -------------------------------')

    #z = A_handle(A_params,x)
    z = x
    z_w, noise_vec = AddNoise(z,sigma_w)
    print('z_w:', z_w)

    # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!! #
    if 1.0 == n_bit:
        quan_step = 2.0
    elif 2.0 == n_bit:
        quan_step = 0.5
    else:
        #quan_step = (tf.reduce_max(input_tensor = tf.abs(x = y_1), axis = 0, keepdims = True) + 1.0) / math.pow(2, n_bit)
        quan_step = 2.0 / np.power(2.0, n_bit)
    print('quan_step:', quan_step)

    tmp1 = np.arange(start = 0.0, stop = np.power(2.0, n_bit - 1.0), step = 1.0, dtype = np.float32)
    tmp1 = np.reshape(a = tmp1, newshape = (tmp1.shape[0], 1))
    print('tmp1:', tmp1)

    tmp2 = np.arange(start = 1.0, stop = np.power(2.0, n_bit), step = 2.0, dtype = np.float32)
    tmp2 = np.reshape(a = tmp2, newshape = (tmp2.shape[0], 1))
    print('tmp2:', tmp2)

    if 1.0 == n_bit or 2.0 == n_bit:
        DeltaTh = tmp1 * quan_step
        Q_out = tmp2 * quan_step / 2.0
    else:
        #DeltaTh = tf.matmul(a = tmp1, b = quan_step)
        #Q_out = tf.matmul(a = tmp2, b = quan_step / 2.0)
        DeltaTh = tmp1 * quan_step
        Q_out = tmp2 * quan_step / 2.0
    print('DeltaTh:', DeltaTh)
    print('Q_out:', Q_out)

    if 1.0 == n_bit or 2.0 == n_bit:
        print('DeltaTh[-1, 0]:', DeltaTh[-1, 0])
        print('Q_out[-1, 0]:', Q_out[-1, 0])
        y_R = Q_out[-1, 0] * tf.cast(x = tf.greater_equal(x = tf.abs(x = z_w), y = DeltaTh[-1, 0]), dtype = tf.float32)
        for bIdx in reversed(range(Q_out.shape[0] - 1)):
            print('bIdx:', bIdx)
            y_R += Q_out[bIdx, 0] * tf.cast(x = tf.greater_equal(x = tf.abs(x = z_w), y = DeltaTh[bIdx, 0]), dtype = tf.float32) * \
                tf.cast(x = tf.less(x = tf.abs(x = z_w), y = DeltaTh[bIdx + 1, 0]), dtype = tf.float32)
    else:
        print('DeltaTh[-1]:', DeltaTh[-1])
        print('Q_out[-1]:', Q_out[-1])
        y_R = Q_out[-1] * tf.cast(x = tf.greater_equal(x = tf.abs(x = z_w), y = DeltaTh[-1]), dtype = tf.float32)
        for bIdx in range(DeltaTh.shape[0] - 1):
            #print('bIdx:', bIdx)
            y_R += Q_out[bIdx] * tf.cast(x = tf.greater_equal(x = tf.abs(x = z_w), y = DeltaTh[bIdx]), dtype = tf.float32) * \
                tf.cast(x = tf.less(x = tf.abs(x = z_w), y = DeltaTh[bIdx + 1]), dtype = tf.float32)

    print('y_R:', y_R)
    y = tf.sign(x = z_w) * y_R
    print('y:', y)

    return quan_step, y

def GenerateNoisyCSData_handles_Ex(x,A_handle,sigma_w,A_params, n_bit):
    print('------------------------------- GenerateNoisyCSData_handles_Ex -------------------------------')

    # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!! #
    if 1.0 == n_bit:
        quan_step = 2.0
    elif 2.0 == n_bit:
        quan_step = 0.5
    else:
        #quan_step = (tf.reduce_max(input_tensor = tf.abs(x = y_1), axis = 0, keepdims = True) + 1.0) / math.pow(2, n_bit)
        quan_step = 2.0 / np.power(2.0, n_bit)
    print('quan_step:', quan_step)

    tmp1 = np.arange(start = 0.0, stop = np.power(2.0, n_bit - 1.0), step = 1.0, dtype = np.float32)
    tmp1 = np.reshape(a = tmp1, newshape = (tmp1.shape[0], 1))
    print('tmp1:', tmp1)

    tmp2 = np.arange(start = 1.0, stop = np.power(2.0, n_bit), step = 2.0, dtype = np.float32)
    tmp2 = np.reshape(a = tmp2, newshape = (tmp2.shape[0], 1))
    print('tmp2:', tmp2)

    if 1.0 == n_bit or 2.0 == n_bit:
        DeltaTh = tmp1 * quan_step
        Q_out = tmp2 * quan_step / 2.0
    else:
        #DeltaTh = tf.matmul(a = tmp1, b = quan_step)
        #Q_out = tf.matmul(a = tmp2, b = quan_step / 2.0)
        DeltaTh = tmp1 * quan_step
        Q_out = tmp2 * quan_step / 2.0
    print('DeltaTh:', DeltaTh)
    print('Q_out:', Q_out)

    if 1.0 == n_bit or 2.0 == n_bit:
        print('DeltaTh[-1, 0]:', DeltaTh[-1, 0])
        print('Q_out[-1, 0]:', Q_out[-1, 0])
        y_R = Q_out[-1, 0] * tf.cast(x = tf.greater_equal(x = tf.abs(x = x), y = DeltaTh[-1, 0]), dtype = tf.float32)
        for bIdx in reversed(range(Q_out.shape[0] - 1)):
            print('bIdx:', bIdx)
            y_R += Q_out[bIdx, 0] * tf.cast(x = tf.greater_equal(x = tf.abs(x = x), y = DeltaTh[bIdx, 0]), dtype = tf.float32) * \
                tf.cast(x = tf.less(x = tf.abs(x = x), y = DeltaTh[bIdx + 1, 0]), dtype = tf.float32)
    else:
        print('DeltaTh[-1]:', DeltaTh[-1])
        print('Q_out[-1]:', Q_out[-1])
        y_R = Q_out[-1] * tf.cast(x = tf.greater_equal(x = tf.abs(x = x), y = DeltaTh[-1]), dtype = tf.float32)
        for bIdx in range(DeltaTh.shape[0] - 1):
            #print('bIdx:', bIdx)
            y_R += Q_out[bIdx] * tf.cast(x = tf.greater_equal(x = tf.abs(x = x), y = DeltaTh[bIdx]), dtype = tf.float32) * \
                tf.cast(x = tf.less(x = tf.abs(x = x), y = DeltaTh[bIdx + 1]), dtype = tf.float32)

    print('y_R:', y_R)
    Q_x = tf.sign(x = x) * y_R
    print('Q_x:', Q_x)


    z = A_handle(A_params, Q_x)
    y, noise_vec = AddNoise(z, sigma_w)
    print('y:', y)

    return quan_step, y

def GenerateNoisyCSData_handles_2(x, sigma_w, n_bit):
    print('------------------------------- GenerateNoisyCSData_handles_Ex -------------------------------')

    # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!! #
    if 1.0 == n_bit:
        quan_step = 2.0
    elif 2.0 == n_bit:
        quan_step = 0.5
    else:
        #quan_step = (tf.reduce_max(input_tensor = tf.abs(x = y_1), axis = 0, keepdims = True) + 1.0) / math.pow(2, n_bit)
        quan_step = 2.0 / np.power(2.0, n_bit)
    print('quan_step:', quan_step)

    tmp1 = np.arange(start = 0.0, stop = np.power(2.0, n_bit - 1.0), step = 1.0, dtype = np.float32)
    tmp1 = np.reshape(a = tmp1, newshape = (tmp1.shape[0], 1))
    print('tmp1:', tmp1)

    tmp2 = np.arange(start = 1.0, stop = np.power(2.0, n_bit), step = 2.0, dtype = np.float32)
    tmp2 = np.reshape(a = tmp2, newshape = (tmp2.shape[0], 1))
    print('tmp2:', tmp2)

    if 1.0 == n_bit or 2.0 == n_bit:
        DeltaTh = tmp1 * quan_step
        Q_out = tmp2 * quan_step / 2.0
    else:
        #DeltaTh = tf.matmul(a = tmp1, b = quan_step)
        #Q_out = tf.matmul(a = tmp2, b = quan_step / 2.0)
        DeltaTh = tmp1 * quan_step
        Q_out = tmp2 * quan_step / 2.0
    print('DeltaTh:', DeltaTh)
    print('Q_out:', Q_out)

    if 1.0 == n_bit or 2.0 == n_bit:
        print('DeltaTh[-1, 0]:', DeltaTh[-1, 0])
        print('Q_out[-1, 0]:', Q_out[-1, 0])
        y_R = Q_out[-1, 0] * tf.cast(x = tf.greater_equal(x = tf.abs(x = x), y = DeltaTh[-1, 0]), dtype = tf.float32)
        for bIdx in reversed(range(Q_out.shape[0] - 1)):
            print('bIdx:', bIdx)
            y_R += Q_out[bIdx, 0] * tf.cast(x = tf.greater_equal(x = tf.abs(x = x), y = DeltaTh[bIdx, 0]), dtype = tf.float32) * \
                tf.cast(x = tf.less(x = tf.abs(x = x), y = DeltaTh[bIdx + 1, 0]), dtype = tf.float32)
    else:
        print('DeltaTh[-1]:', DeltaTh[-1])
        print('Q_out[-1]:', Q_out[-1])
        y_R = Q_out[-1] * tf.cast(x = tf.greater_equal(x = tf.abs(x = x), y = DeltaTh[-1]), dtype = tf.float32)
        for bIdx in range(DeltaTh.shape[0] - 1):
            #print('bIdx:', bIdx)
            y_R += Q_out[bIdx] * tf.cast(x = tf.greater_equal(x = tf.abs(x = x), y = DeltaTh[bIdx]), dtype = tf.float32) * \
                tf.cast(x = tf.less(x = tf.abs(x = x), y = DeltaTh[bIdx + 1]), dtype = tf.float32)

    print('y_R:', y_R)
    Q_x = tf.sign(x = x) * y_R
    print('Q_x:', Q_x)


    #z = A_handle(A_params, Q_x)
    z = Q_x
    y, noise_vec = AddNoise(z, sigma_w)
    print('y:', y)
    #return quan_step, y
    return quan_step, z, y

## Create training data from images, with tf
def GenerateNoisyCSData(x,A,sigma_w):
    y = tf.matmul(A,x)
    y = AddNoise(y,sigma_w)
    return y

## Create training data from images, with tf
def AddNoise(clean,sigma):
    if is_complex:
        noise_vec = sigma/np.sqrt(2) *( tf.complex(tf.random_normal(shape=clean.shape, dtype=tf.float32),tf.random_normal(shape=clean.shape, dtype=tf.float32)))
    else:
        noise_vec=sigma*tf.random_normal(shape=clean.shape,dtype=tf.float32)

    noisy=clean+noise_vec
    noisy=tf.reshape(noisy,clean.shape)
    #return noisy
    return noisy, noise_vec

## Create training data from images, with numpy
def GenerateNoisyCSData_np(x,A,sigma_w):
    y = np.matmul(A,x)
    y = AddNoise_np(y,sigma_w)
    return y

## Create training data from images, with numpy
def AddNoise_np(clean,sigma):
    noise_vec=np.random.randn(clean.size)
    noise_vec = sigma * np.reshape(noise_vec, newshape=clean.shape)
    noisy=clean+noise_vec
    return noisy

##Create a string that generates filenames. Ensures consitency between functions
def GenLDGAMP_ResNetFilename(alg,tie_weights,LayerbyLayer,n_GAMP_layer_override=None,sampling_rate_override=None,loss_func='MSE'):
    if n_GAMP_layer_override:
        n_GAMP_layers_save=n_GAMP_layer_override
    else:
        n_GAMP_layers_save=n_GAMP_layers
    if sampling_rate_override:
        sampling_rate_save=sampling_rate_override
    else:
        sampling_rate_save=sampling_rate
    if loss_func=='SURE':
        filename = "./saved_models/LD_GAMP_R/SURE_"+alg+"_" + str(num_layers_in_ed_block) + "ResNetL_" + str(int(n_GAMP_layers_save)) + "GAMPL_Tie"+str(tie_weights)+"_LbyL"+str(LayerbyLayer)+"_SR" +str(int(sampling_rate_save*100))
    elif loss_func=='GSURE':
        filename = "./saved_models/LD_GAMP_R/GSURE_"+alg+"_" + str(num_layers_in_ed_block) + "ResNetL_" + str(int(n_GAMP_layers_save)) + "GAMPL_Tie"+str(tie_weights)+"_LbyL"+str(LayerbyLayer)+"_SR" +str(int(sampling_rate_save*100))
    else:
        filename = "./saved_models/LD_GAMP_R/"+alg+"_ResNetL_" + str(int(n_GAMP_layers_save)) + "GAMPL_Tie"+str(tie_weights)+"_LbyL"+str(LayerbyLayer)+"_SR" +str(int(sampling_rate_save*100))
    return filename

##Create a string that generates filenames. Ensures consitency between functions
def GenResNetFilename(sigma_w_min,sigma_w_max,useSURE=False):
    if useSURE:
        filename = "./saved_models/ResNet/SURE_ResNet_sigmaMin" + str(
            int(255. * sigma_w_min)) + "_sigmaMax" + str(int(255. * sigma_w_max))
    else:
        filename="./saved_models/ResNet/ResNet_sigmaMin" + str(int(255.*sigma_w_min))+"_sigmaMax" + str(int(255.*sigma_w_max))
    return filename

## Count the total number of learnable parameters
def CountParameters():
    total_parameters = 0
    for variable in tf.trainable_variables():
        shape = variable.get_shape()
        variable_parameters = 1
        for dim in shape:
            variable_parameters *= dim.value
        total_parameters += variable_parameters
    print('Total number of parameters: ')
    print(total_parameters)

## Calculate Monte Carlo SURE Loss
def MCSURE_loss(x_hat,div_overN,y,sigma_w):
    return tf.reduce_sum(tf.reduce_sum((y - x_hat) ** 2, axis=0) / n_fp -  sigma_w ** 2 + 2. * sigma_w ** 2 * div_overN)

## Calculate Monte Carlo Generalized SURE Loss (||Px||^2 term ignored below)
def MCGSURE_loss(x_hat,x_ML,P,MCdiv,sigma_w):
    Pxhatnorm2 = tf.reduce_sum(tf.abs(tf.matmul(P, x_hat)) ** 2, axis=0)
    temp0 = tf.multiply(x_hat, x_ML)
    x_hatt_xML = tf.reduce_sum(temp0, axis=0)  # x_hat^t*(A^\dagger y)
    return tf.reduce_sum(Pxhatnorm2+2.*sigma_w**2*MCdiv-2.*x_hatt_xML)

## Calculate Monte Carlo Generalized SURE Loss, ||Px||^2 explicitly added so that estimate tracks MSE
def MCGSURE_loss_oracle(x_hat,x_ML,P,MCdiv,sigma_w,x_true):
    Pxtruenorm2 = tf.reduce_sum(tf.abs(tf.matmul(P, x_true)) ** 2, axis=0)
    return tf.reduce_sum(Pxtruenorm2)+MCGSURE_loss(x_hat,x_ML,P,MCdiv,sigma_w)