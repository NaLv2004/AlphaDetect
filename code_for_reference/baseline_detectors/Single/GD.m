function symest = GD(TxAntNum,RxAntNum,RxSymbol,Hreal,sym,iterNum,Nv)

%% Nestrerov's accelerated gradient descent algorithm

lambda= 1/norm(Hreal'*Hreal,'fro');
beta=0.9;

Tx=zeros(2*TxAntNum,1);
d=zeros(2*TxAntNum,1);

for i=1:40

    Tx_new=Tx-beta*d;
    grad= Hreal'*(Hreal*Tx_new-RxSymbol)+Nv*Tx;

    d=beta*d+lambda*grad;
    Tx=Tx- d;

end

[~,index]=min(abs(Tx-sym'),[],2);
symest=sym(index);

 
%% Adam

% lambda= 1/norm(Hreal'*Hreal,'fro');
% beta1=0.9;
% beta2=0.999;
% epsilon=1e-8;
% 
% Tx=zeros(2*TxAntNum,1);
% m=zeros(2*TxAntNum,1);
% v=zeros(2*TxAntNum,1);
% 
% for i=1:iterNum
% 
%     grad= Hreal'*(Hreal*Tx-RxSymbol);
%     m=beta1*m+(1-beta1)*grad;
%     v=beta2*v+(1-beta2)*grad.^2;
%     m_hat=m/(1-beta1^i);
%     v_hat=v/(1-beta2^i);
%     Tx=Tx-lambda*m_hat./(sqrt(v_hat)+epsilon);
% 
% end
% 
% [~,index]=min(abs(Tx-sym'),[],2);
% symest=sym(index);


end

