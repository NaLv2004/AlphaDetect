function symest = fista(TxAntNum,RxAntNum,slen,RxSymbol,H,Nv,sym,delta,iterNum,TxSymbol_real)
mu=100000;

B=repmat(1:2*TxAntNum*slen,2*TxAntNum,1)-(0:(2*TxAntNum-1))'*slen;
B=B<=slen & B>=1;
S=repmat(sym',2*TxAntNum,2*TxAntNum).* B;

H_reg=[H*S;mu*B];
y_reg=[RxSymbol;mu*ones(2*TxAntNum,1)];


% px init
px=ones(2*TxAntNum*slen,1)/slen;
x_old=px;
y_old=px;
t_old=1;
iter=0;

HTH=H_reg'*H_reg;
HTY=H_reg'*y_reg;
L=max(eig(HTH));
Linv = 1/L;  
lambda = 10;
lambdaLiv = lambda*Linv;


while iter<1000000
    iter=iter+1;
    x_new=y_old -Linv*(HTH*y_old-HTY);
    x_new = max(0, x_new - lambdaLiv) + min(0, x_new + lambdaLiv);
    t_new = 0.5*(1 + sqrt(1 + 4*t_old^2));
    y_new = x_new + (t_old - 1)/t_new * (x_new - x_old);

    x_old = x_new;
    t_old = t_new;
    y_old = y_new;



end


S_mmse = (H'*H+Nv*eye(2*TxAntNum)) \ H'*RxSymbol;
[~,Indice]=min(abs(S_mmse-sym'),[],2);
symest = sym(Indice);


end
