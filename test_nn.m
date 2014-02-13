function y = test_nn(w1,w2,phi_test)

T_test=zeros(size(phi_test,1),10);

for i=1:size(phi_test,1)
    j=phi_test(i,513);
    T_test(i,j+1)=1;
end

phi_test=[ones(size(phi_test,1),1) phi_test(:,1:512)];

a_j=phi_test*(w1);

z_j=1./(1+exp(-a_j));

a_k=z_j*(w2);

exp_ak=exp(a_k);

sum_exp_ak=sum(exp_ak,2);

output=zeros(1500,10);

for i=1:1500
    for j=1:10
        output(i,j)=exp_ak(i,j)./sum_exp_ak(i,1);
    end
end

%ind2sub(output,max(output,[],2));
[~,num]=max(output,[],2);

prediction=zeros(1500,10);

for i=1:1500
     j=num(i,1);
     prediction(i,j)=1;
end

%Calculate Error Rate and Reciprocal Rank
miss=(sum(sum(abs(T_test-prediction))))/2;
error_rate=(miss/size(phi_test,1)).*100;
for RR=1:size(phi_test,1)
    if T_test(RR,:)==prediction(RR,:)
        break;
    end
end

[~,T_label]=max(prediction,[],2);
T_label=T_label-1;

fprintf('the Error Rate for the Neural Network model is %d', error_rate ); 
        fprintf('\n');
fprintf('the Reciprocal Rank for the Neural Network model is %d', 1/RR ); 
        fprintf('\n');
        
y=T_label;
