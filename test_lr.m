function y = test_lr(w,phi_test)

% for i=1:1500
% j=new_test_array(i,513);
% T_test(i,j+1)=1;
% end
% phi_test=[test_0;test_1;test_2;test_3;test_4;test_5;test_6;test_7;test_8;test_9;];

T_test=zeros(size(phi_test,1),10);

for i=1:size(phi_test,1)
    j=phi_test(i,513);
    T_test(i,j+1)=1;
end

phi_test=[ones(size(phi_test,1),1) phi_test(:,1:512)];

%Calculate Output by multiplying weights with Test features.
output=phi_test*w;

%Find the maximum value in each row using the ind2sub function.
%ind2sub(output,max(output,[],2));
[~,num]=max(output,[],2);

%Create the Output Prediction matrix.
prediction=zeros(1500,10);
for i=1:1500
     j=num(i,1);
     prediction(i,j)=1;
end

%Calculate Error Rate and Reciprocal Rank.
miss=(sum(sum(abs(T_test-prediction))))/2;
error_rate=(miss/size(phi_test,1)).*100;
for RR=1:size(phi_test,1)
    if T_test(RR,:)==prediction(RR,:)
        break;
    end
end

[~,T_label]=max(prediction,[],2);
T_label=T_label-1;

fprintf('the Error Rate for the Multi Class Logistic Regression model is %d', error_rate ); 
        fprintf('\n');
fprintf('the Reciprocal Rank for the Multi Class Logistic Regression model is %d', 1/RR ); 
        fprintf('\n');
        
y=T_label;
        