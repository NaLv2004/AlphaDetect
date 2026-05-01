function symest = MMSE(TxAntNum,RxAntNum,slen,RxSymbol,H,Nv,sym,delta,iterNum)
S_mmse = (H'*H+Nv*eye(2*TxAntNum)) \ H'*RxSymbol;
[~,Indice]=min(abs(S_mmse-sym'),[],2);
symest = sym(Indice);


end

