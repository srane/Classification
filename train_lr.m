function w = train_lr(phi)

%clear all
%close all
%load('data.mat');
%phi=[train{1};train{2};train{3};train{4};train{5};train{6};train{7};train{8};train{9};train{10}];

T=zeros(size(phi,1),10);

for i=1:size(phi,1)
    j=phi(i,513);
    T(i,j+1)=1;
end

%INITIALIZATION
phi=[ones(size(phi,1),1) phi(:,1:512)];
K=10;%Number of Output classes
D=512;%Number of Features
w_old=zeros(D+1,K);
w_star=zeros(D+1,K);
old_error=inf;
gradient_count=0;%Number of Iterations of Gradient Descent
stepper=0;%To change Step Size
step=0.0001;%Step size
err_struct=struct('f',0);%For graph of Error
error=inf;

%Iterate as long as error is above 2 and Iterations are less than 50000.
while((old_error>2&&~isnan(error))&&gradient_count<50000)

    %a is the activation function.
    a=phi*w_old;

%This is the Soft-Max function.
y=zeros(19978,10);
exp_a=exp(a);
exp_sum=sum(exp_a,2);
for i=1:19978
    for j=1:10
     y(i,j)=exp_a(i,j)./exp_sum(i,1);
    end
end

%Calculate delta E
delta_E=(phi'*(y-T));

%Gradient Descent for updating weight values.
w_new=w_old-(step.*delta_E);
gradient_count=gradient_count+1;

%Cross Entropy Error
error=-sum(sum(T.*log(y)));
err_struct(gradient_count).f=error;
display(error);

%VARIABLE STEP SIZE CALCULATION CODE
if old_error<=error %If no change in Error, maintain small step size
    stepper=0;
    step=0.0001;
    
    else
    stepper=stepper+1;%If error decreases 5 times consecutively, increment step size.
    if stepper>5
        step=step+0.0005;
        stepper=0;
    end
    
    
    w_star=w_new;%Store best Weight values.
    old_error=error;
   
end
w_old=w_new;
end
w=w_star;