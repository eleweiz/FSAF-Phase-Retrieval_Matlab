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

%% Load image
mx = 1080;
my = 1920;
E_tem1=imread('galaxy.jpg');  % from "Pics about space"
for kk=1:3
    E_tem2=E_tem1(:,:,kk);
    E_tem2=double(E_tem2);
    E_tem(:,:,kk) =double(imresize(E_tem2,[mx my]));
end
Em=E_tem;
% figure;
% imshow(mat2gray(abs(Em)),[]);

%% Make masks and linear sampling operators
L = 3;                  % Number of masks
Masks = zeros(mx,my,L);  % Storage for L masks, each of dim n1 x n2
for ll = 1:L, Masks(:,:,ll) = randsrc(mx,my,[1i -1i 1 -1]); end
A = @(I)  fft2(conj(Masks) .* reshape(repmat(I,[1 L]), size(I,1), size(I,2), L));  % Input is n1 x n2 image, output is n1 x n2 x L array
At = @(Y) mean(Masks .* ifft2(Y), 3);                                              % Input is n1 x n2 X L array, output is n1 x n2 image
S = @(I,l)  fft2(conj(Masks(:,:,l)) .* I);  % Input is n1 x n2 image, output is n1 x n2 array
St = @(Y,l) Masks(:,:,l) .* ifft2(Y);    % Input is n1 x n2 array, output is n1 x n2 image

for rgb = 1:3
    % Data
    fprintf('Color band %d\n', rgb)
    x = squeeze(Em(:,:,rgb)); % Image x is n1 x n2
    Y = abs(A(x)).^2;        % Measured data
    
    %% Low Freq Initialization
    M0=30;
    npower_iter = 20;                          % Number of power iterations
    Alpha = randn(M0,M0);
    Alpha = Alpha/norm(Alpha,'fro'); % Initial guess
    alpha=2.3;     % Threshold factor
    normest = sqrt(sum(Y(:))/numel(Y(:)));    % Estimate norm to scale eigenvector
    flag1=abs(Y)<=alpha^2 * normest^2 ;  % Truncation
    T_ration=size(find(1-flag1))/(mx*my);
    Ytr = Y.* flag1;
    % Power iterations
    for tt = 1:npower_iter
        Alpha_tem1=zeros(mx,my);
        Alpha_tem1(0.5*(mx-M0)+1:0.5*(mx-M0)+M0,0.5*(my-M0)+1:0.5*(my-M0)+M0)=Alpha;
        z0=fft2(ifftshift(Alpha_tem1));
        z1=Ytr.*A(z0);
        Alpha_tem2=fftshift(ifft2(At(z1)));
        Alpha=Alpha_tem2(0.5*(mx-M0)+1:0.5*(mx-M0)+M0,0.5*(my-M0)+1:0.5*(my-M0)+M0);
        Alpha = Alpha/norm(Alpha,'fro');
    end
    Alpha_tem1=zeros(mx,my);
    Alpha_tem1(0.5*(mx-M0)+1:0.5*(mx-M0)+M0,0.5*(my-M0)+1:0.5*(my-M0)+M0)=Alpha;
    z_tem=fft2(ifftshift(Alpha_tem1));
    z_tem=z_tem/norm(z_tem,'fro');
    normest = sqrt(sum(Y(:))/numel(Y)); % Estimate norm to scale eigenvector
    z = normest * z_tem;                   % Apply scaling
    Relerrs = norm(x - exp(-1i*angle(trace(x'*z))) * z, 'fro')/norm(x,'fro'); % Initial rel. error
       
    %% Loop
    gamma_t=0.7;
    M = numel(sqrt(Y));
    T=50;        % Iterations
    tau0 = 330;                         % Time constant for step size
    mu = @(t) min(1-exp(-t/tau0), 0.4); % Schedule for step size
    for t = 1:T,
        Bz = A(z);
        % C=Bz - sqrt(Y) .* exp(1i * angle(Bz));
        %  grad = At(C);  % Wirtinger gradient
        ind = (abs(Bz)./ sqrt(Y) >= 1 / (1 + gamma_t)); % gradient truncation
        grad = At(ind .* (Bz - sqrt(Y) .* exp(1i * angle(Bz)))) / M;
        grad=grad(:);
        
        %% Conjugate gradient (Can also be driectly used to Wirtinger gradient)
        if(t==1)
            d_prp=-grad;
        else
            betak=real(grad'*(grad-g0))/(g0'*g0);   %% g stands for g(k+1), g0 stands for g(k)
            d_prp=-grad+betak*d0;
        end
        d=-reshape(d_prp,mx,my);
        %% Optimal size  (Can also be driectly used to Wirtinger gradient)
        h_i=A(d);
        u_i=real(conj(Bz).*h_i);
        r_i=abs(Bz).^2-Y;
        a_c=sum(sum(sum(abs(h_i).^4)));
        b_c=-3*sum(sum(sum(u_i.*abs(h_i).^2)));
        c_c=sum(sum(sum(r_i.*abs(h_i).^2+2*u_i.^2)));
        d_c=-sum(sum(sum(u_i.*r_i)));
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
            M_f=[norm(mean(abs(A(z1)).^2-Y,3),'fro')/norm(mean(Y,3),'fro'),norm(mean(abs(A(z2)).^2-Y,3),'fro')/norm(mean(Y,3),'fro'),norm(mean(abs(A(z3)).^2-Y,3),'fro')/norm(mean(Y,3),'fro')];
            Indx1=find(M_f==min(M_f));
            Step(t)=r(Indx1);
        else
            Step(t)=mu(t)/normest^2;
        end
        
        z = z-Step(t) * d;  % Gradient update
        g0=grad; d0=d_prp;
        Relerrs = [Relerrs, norm(x - exp(-1i*angle(trace(x'*z))) * z, 'fro')/norm(x,'fro')];
    end
    Err(rgb,:)=Relerrs;
    X(:,:,rgb)=z;
end
fprintf('All done!\n')
%%  Check results
fprintf('Relative error after initialization: %f\n', Relerrs(1))

fprintf('Relative error after %d iterations: %f\n', T, Relerrs(T+1))
figure;
imshow(mat2gray(abs(X)),[]);title('Reconstruction');
for kk=1
    figure;
    semilogy(Err(kk,:),'ks-');title('error');
end



