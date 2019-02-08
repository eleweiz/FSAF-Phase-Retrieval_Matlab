%% Copyright © 2018,  National University of Singapore, Zhun Wei (weizhun1010@gmail.com)
% Implementation of the Frequency Subspace Amplitude Flow algorithm proposed in the paper
%  ``Frequency Subspace Amplitude Flow for Phase
% Retrieval'' by Z. Wei, W. Chen, and X. Chen (2018)
% The code below is adapted from implementation of the Wirtinger Flow
% algorithm designed and implemented by E. J. Candes, X. D. Li, and M. Soltanolkotabi

% The code use Frequency subspace to reduce the sample complexity at initialization stage;
% The code also use conjugate gradient and optimal stepsize to accelerate the convergence
% rate, where both of them can also be used to Wirtinger Flow. The conjugate strategy is
% proposed in ``Conjugate gradient method for phase retrieval based on the Wirtinger 
% derivative'' by Z. Wei, W. Chen, C.-W. QIU, and X. Chen (2017);
clc;clear all;close all;

L_M=2.6;    % m/n
n=500;
x_tem1=(linspace(1,2*pi,n)).';
% x_real=sin(2*x_tem1).*cos(1*x_tem1);x_imag=sin(3*x_tem1).*cos(2*x_tem1); % Sin signal
x_real=rand(n,1);x_imag=rand(n,1); % Uniformly distributed
x=x_real+i*x_imag;
m = round(L_M*n);
A = 1/sqrt(2)*randn(m,n) + 1i/sqrt(2)*randn(m,n);
%% Data
Y0 = abs(A*x).^2;
var= 0.00; % noiseless would correspond to var = 0
P1=size(Y0,1);Q1=size(Y0,2);
rand_real = randn([P1,Q1]);
y_Gaussian = sqrt(1/P1/Q1)*norm(sqrt(Y0),'fro') *var*rand_real;   % check the value
NSR=norm(y_Gaussian,'fro')/norm(sqrt(Y0),'fro')
Y = abs(A*x + y_Gaussian).^2;

%% Low Freq Initialization
M0=10;
npower_iter = 20;                          % Number of power iterations 
Alpha = randn(M0,1);
Alpha = Alpha/norm(Alpha,'fro'); % Initial guess 
alpha=2.3;     % Threshold factor
normest = sqrt(sum(Y(:))/numel(Y(:))); % Estimate norm to scale eigenvector
flag1=abs(Y)<=alpha^2 * normest^2 ;  % Truncation
T_ration=size(find(1-flag1))/(n);
Ytr = Y.* flag1;
% Power iterations 
for tt = 1:npower_iter
    Alpha_tem1=zeros(n,1);
    Alpha_tem1(0.5*(n-M0)+1:0.5*(n-M0)+M0,1)=Alpha;
    z0=fft(ifftshift(Alpha_tem1));
    z1=Ytr.*(A*z0);
    Alpha_tem2=fftshift(ifft(A'*z1));
    Alpha=Alpha_tem2(0.5*(n-M0)+1:0.5*(n-M0)+M0,1);
    Alpha = Alpha/norm(Alpha,'fro');
end
Alpha_tem1=zeros(n,1);
Alpha_tem1(0.5*(n-M0)+1:0.5*(n-M0)+M0,1)=Alpha;
z_tem=fft(ifftshift(Alpha_tem1));
z_tem=z_tem/norm(z_tem,'fro');
normest = sqrt(sum(Y(:))/numel(Y)); % Estimate norm to scale eigenvector  
z = normest * z_tem;                   % Apply scaling 
Relerrs = norm(x - exp(-1i*angle(trace(x'*z))) * z, 'fro')/norm(x,'fro'); % Initial rel. error

%% Loop
T = 100;                           % Iterations
tau0 = 330;                         % Time constant for step size
Coef1=5;
mu = @(t) min(1-exp(-t/tau0), 0.2); % Schedule for step size
% beta=0.4;
for t = 1:T,
    
    yz = A*z;
    ratio = abs(yz) ./ sqrt(Y);
    Tru_g= ratio > 1 / (1 + 0.7);
    C=(yz - sqrt(Y) .* exp(1i * angle(yz))).*Tru_g;  % gradient truncation
    grad  = 1/m* A'*C; % Wirtinger gradient
    %% Conjugate gradient (Can also be driectly used to Wirtinger gradient)
    if(t==1)
        d_prp=-grad;
    else
        betak=real(grad'*(grad-g0))/(g0'*g0);   %% g stands for g(k+1), g0 stands for g(k)
        d_prp=-grad+betak*d0;
    end
    d=-d_prp;  
    %% Optimal size  (Can also be driectly used to Wirtinger gradient)
    h_i=A*d;
    u_i=real(conj(yz).*h_i);
    r_i=abs(yz).^2-Y;
    a_c=sum(abs(h_i).^4);
    b_c=-3*sum(u_i.*abs(h_i).^2);
    c_c=sum(r_i.*abs(h_i).^2+2*u_i.^2);
    d_c=-sum(u_i.*r_i);
    % Solution of cubic equation
    p=[a_c b_c c_c d_c];
    r=roots(p);
    res=find(~imag(r));
    N_re(t)=size(res,1);
    if N_re(t)==1
        Step(t)=r(res);
    elseif N_re(t)==3
        L_f=[3*a_c*r(1)^2+2*b_c*r(1)+c_c,3*a_c*r(2)^2+2*b_c*r(2)+c_c,3*a_c*r(3)^2+2*b_c*r(3)+c_c];
        z1=z-r(1)*d; z2=z-r(2)*d;z3=z-r(3)*d;
        M_f=[norm(abs(A*z1).^2-Y,'fro')/norm(Y,'fro'),norm(abs(A*z2).^2-Y,'fro')/norm(Y,'fro'),norm(abs(A*z3).^2-Y,'fro')/norm(Y,'fro')];
        Indx1=find(M_f==min(M_f));
        Step(t)=r(Indx1);
    else
        Step(t)=mu(t)/normest^2;
    end
    
    z=z-Step(t)*d;
    g0=grad; d0=d_prp;
    
    Relerrs = [Relerrs, norm(x - exp(-1i*angle(trace(x'*z))) * z, 'fro')/norm(x,'fro')];
end

figure;
semilogy(0:2:T,Relerrs(1:2:end),'k^-');title('Error');

figure
hold on
plot(abs(x),'b*');
plot(abs(z),'r-');
legend('GT','Rec');
title('Amplitude');
figure
hold on
plot(angle(x),'b*');
plot(angle(exp(-1i*angle(trace(x'*z))) * z),'r-');
legend('GT','Rec')
title('Phase');









