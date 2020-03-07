%function[hat_x,hat_v]=CGAMP(obj,Input)
function MSE_error=GAMP_Iter(obj,Input)

IterNum=Input.IterNum;
kk=Input.kk;
N=Input.N;
M=Input.M;
H=obj.H;
%xo=obj.xo;
%mes=Input.mes;

%%arrayinitialization
%hat_x=zeros(N,1);

%hat_x=sqrt(double(trace(obj.y*obj.y')/double(N)-Input.nuw*Input.nuw))*ones(N,1);

%symsx_sys;
%pdf=1.0e+04*(-6.367503924646788*x_sys^2+6.351405706898148*x_sys+(-0.135662027149322));

%mean_sys=int(pdf*x_sys,x_sys,0,1);
%var_sys=int(pdf*(x_sys-mean_sys)^2,x_sys,0,1);

%mean_val=double(subs(mean_sys));
%var_val=double(subs(var_sys));

%hat_x=mean_val*ones(N,1);
%hat_v=var_val*ones(N,1);

hat_x=zeros(N,1);
hat_v=ones(N,1);

hat_s=zeros(M,1);

sqrH=abs(H).^2;
sqrHt=sqrH';
Ht=H';
%V_old=ones(M,1);
%Z_old=zeros(M,1);

%hat_x_old=0;
%hat_v_old=0;

MSE_error=zeros(1,IterNum);

for ii=1:IterNum

%%OutputNodes
V=sqrH*hat_v;
Z=H*hat_x-hat_s.*V;
%[Z,Z_old]=damping(Z,Z_old,mes);
%[V,V_old]=damping(V,V_old,mes);
[hat_s,hat_tau]=estimator_2(Z,V,obj,Input);

%%InputNodes
Sigma=(sqrHt*hat_tau).^(-1);
R=hat_x+Sigma.*(Ht*hat_s);

%[hat_x,hat_v]=estimator_1(xo,R,Sigma);

[hat_x,hat_v]=estimator_1(R,Sigma);

save(['./kk_',num2str(kk),'_IterNum_',num2str(ii),'_hat_x.mat'],'hat_x')
MSE_error(1,ii)=norm(hat_x-obj.x).^2/double(N);
if sum(isnan(hat_v))>0
%hat_x=hat_x_old;
%hat_v=hat_v_old;
%print'sum(isnan(hat_v))>0'
1111111111111111111111111111111111111111111111111111111111111111111111111111111
break;
end
%hat_x_old=hat_x;
%hat_v_old=hat_v;
end
end

function [x,x_old]=damping(x,x_old,mes)
x=mes*x+(1-mes)*x_old;
x_old=x;
end

%function[m,v]=estimator_1(xo,check_m,check_v)
function[m,v]=estimator_1(check_m,check_v)
%log_posterior=bsxfun(@times,-1./check_v,abs(bsxfun(@minus,xo,check_m).^2));
%log_posterior=bsxfun(@minus,log_posterior,max(log_posterior));%��ֹ��ﹿ

%posterior=exp(log_posterior);
%posterior=bsxfun(@rdivide,posterior,sum(posterior,2));%�õ���׼PDF
%m=sum(bsxfun(@times,posterior,xo),2);%����PDF�ľ�ֵ
%v=sum(posterior.*abs(bsxfun(@minus,m,xo).^2),2);%����PDF�ķ���

m=zeros(size(check_m));
v=zeros(size(check_m));

syms x_sys v_sys m_sys;

%tmp0=1.0e+07*(1.1023*x_sys^3+(-1.8557)*x_sys^2+0.7938*x_sys^1+0.0153*x_sys^0);
%tmp0=1.0e+04*(-6.367503924646788*x_sys^2+6.351405706898148*x_sys+(-0.135662027149322));
%f = abs((-0.0570)*x_sys^3+(-0.0533)*x_sys^2+(0.1070)*x_sys^1+(-0.0015)*x_sys^0);
%c_f = int(f, x_sys, 0, 1);
%tmp0 = f / c_f;
tmp0 = 1.2444*x_sys^6+(-2.2477)*x_sys^5+1.0243*x_sys^4+(-0.0088)*x_sys^3+(-0.0897)*x_sys^2+0.0862*x_sys^1+0.0010;

tmp1=int(tmp0*x_sys*exp(-(x_sys-m_sys)^2/(2*v_sys)),x_sys,0,1);
tmp2=int(tmp0*exp(-(x_sys-m_sys)^2/(2*v_sys)),x_sys,0,1);
eta=tmp1/tmp2;
tmp3=int(tmp0*x_sys^2*exp(-(x_sys-m_sys)^2/(2*v_sys)),x_sys,0,1);
theta=tmp3/tmp2-eta^2;

for i=1:size(check_m,1)
m(i)=double(subs(eta,[m_sys,v_sys],[check_m(i),check_v(i)]));
v(i)=double(subs(theta,[m_sys,v_sys],[check_m(i),check_v(i)]));
end
end

function[hat_s,hat_tau]=estimator_2(Z,V,obj,Input)
nuw=Input.nuw;
%AGC_switch=Input.AGC_switch;
y=obj.y;
M=Input.M;

%ifAGC_switch==0
%hat_s=(y-Z)./(V+nuw);
%hat_tau=1./(V+nuw);
%else
quan_step=obj.quan_step;
Quan_bound=(2^(Input.bit-1)-1)*quan_step;
%y=[real(y);imag(y)];
%z=[real(Z);imag(Z)];
z=Z;
%v=[real(V);real(V)];
v=V;

threhold=10*Quan_bound;

y_up=y+quan_step/2;
y_low=y-quan_step/2;
[pos1,~]=find(y_up>Quan_bound);
[pos2,~]=find(y_low<-Quan_bound);
%y_up(pos1)=1e5;
%y_low(pos2)=-1e5;
y_up(pos1)=10*threhold;
y_low(pos2)=-10*threhold;

%eta1=(sign(y).*z-min(abs(y_up),abs(y_low)))./sqrt((nuw+v)/2);
eta1=(sign(y).*z-min(abs(y_up),abs(y_low)))./sqrt((nuw*nuw+v)/2);
%eta2=(sign(y).*z-max(abs(y_up),abs(y_low)))./sqrt((nuw+v)/2);
eta2=(sign(y).*z-max(abs(y_up),abs(y_low)))./sqrt((nuw*nuw+v)/2);

tem1=normpdf(eta1)-normpdf(eta2);
tem2=normcdf(eta1)-normcdf(eta2);
tem3=eta1.*normpdf(eta1)-eta2.*normpdf(eta2);

%pos=eta2<-100;
%tem1(pos)=normpdf(eta1(pos));
%tem2(pos)=normcdf(eta1(pos));
%tem3(pos)=eta1(pos).*normpdf(eta1(pos));

eps=1e-15;

%z_tem=z+(sign(y).*v./sqrt(2*(nuw+v))).*(tem1./tem2);
z_tem=z+(sign(y).*v./sqrt(2*(nuw*nuw+v))).*(tem1./(tem2+eps));
%v_tem=v/2-((v.^2)./(2*(nuw+v))).*(tem3./tem2+(tem1./tem2).^2);
v_tem=v/2-((v.^2)./(2*(nuw*nuw+v))).*(tem3./(tem2+eps)+(tem1./(tem2+eps)).^2);

%hatz=z_tem(1:M)+1j*z_tem(M+1:2*M);
hatz=z_tem;
%hatv=max(v_tem(1:M)+v_tem(M+1:2*M),1e-10);
hatv=v_tem;

hat_s=(hatz-Z)./V;
hat_tau=-(hatv-V)./(V.^2);

%end
end