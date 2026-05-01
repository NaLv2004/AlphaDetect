function symest = AltMin(TxAntNum,RxAntNum,slen,RxSymbol,H,Nv,sym,delta,iterNum,TxSymbol_real)


x=zeros(2*TxAntNum,1);
lambda=RxSymbol-H*x;
y=H.*x'+lambda/2;

for iter=1:iterNum
    x=sum(y.*H,1)'./sum(H.^2,1)';
    lambda=RxSymbol-H*x;
    y=H.*x'+lambda/2;
end

[~,Indice]=min(abs(x-sym'),[],2);
symest = sym(Indice);


end