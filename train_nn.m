function [w1,w2]=train_nn(phi)

%clear all
%close all
%load ('data.mat');
%phi=[train{1};train{2};train{3};train{4};train{5};train{6};train{7};train{8};train{9};train{10}];
%phi=[ones(19978,1) phi];

T=zeros(size(phi,1),10);

for i=1:size(phi,1)
    j=phi(i,513);
    T(i,j+1)=1;
end

%INITIALIZATION
phi=[ones(size(phi,1),1) phi(:,1:512)];
gradient_count=0;%Number of Iterations
M=(2./3).*513;%Number of Hidden Nodes
K=10;%Number of Output Classes
D=512;%Number of Features
y=zeros(19978,10);
err_struct=struct('f',0);
old_error=inf;
error=inf;
step2=0.0000001;%Step size for W_kj
step1=0.0001;%Step size for W_ji
stepper=0;%For variable Step Size

%Initialize weight values.
w1_old=rand(D+1,M)-0.5;
w2_old=rand(M,K)-0.5;
w1_new=w1_old;
w2_new=w2_old;
w1_star=zeros(D+1,M);
w2_star=zeros(M,K);

%Iterate as long as error is above 0.5 and Iterations are less than 50000.
while((old_error>0.5&&~isnan(error))&&gradient_count<50000)

a_j=phi*(w1_old);

z_j=1./(1+exp(-a_j));

a_k=z_j*(w2_old);

exp_ak=exp(a_k);

sum_exp_ak=sum(exp_ak,2);

for i=1:19978
    for j=1:10
        y(i,j)=exp_ak(i,j)./sum_exp_ak(i,1);
    end
end

%Calculate Cross Entropy Error.
error=-sum(sum(T.*log(y)));
gradient_count=gradient_count+1;
err_struct(gradient_count).f=error;

%Code to Calculate Variable Step Size
if old_error<=error
    stepper=0;
    step2=0.0000001;
    step1=0.0001;
else
    stepper=stepper+1;
    if stepper>5
        step2=step2+0.0000005;
        step1=step1+0.0005;
        stepper=0;
    end

    w1_star=w1_new;
    w2_star=w2_new;
    old_error=error;
    
end

delta_k=y-T;
delta_j=(z_j.*(1-z_j)).*(delta_k*w2_old');

delta_E_2=z_j'*delta_k;
delta_E_1=phi'*delta_j;

%Gradient Descent and Updating Weight Values.
w2_new=w2_old-(step2.*delta_E_2);
w1_new=w1_old-(step1.*delta_E_1);

w2_old=w2_new;
w1_old=w1_new;

display(error);

end
w1=w1_star;
w2=w2_star;
