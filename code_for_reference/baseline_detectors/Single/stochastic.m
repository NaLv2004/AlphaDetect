function symest = stochastic(TxAntNum,RxAntNum,slen,RxSymbol,H,Nv,sym,delta,iterNum,TxSymbol_real)

W=(H'*H+Nv*eye(2*TxAntNum)) \ H';

rho= 2;

S_mmse = W*RxSymbol;

Li=4;
Li_current=0;

Ld=TxAntNum;
Ld_current=0;


[~,Indice]=min(abs(S_mmse-sym'),[],2);
List = sym(Indice);


while Li_current<Li
    v=randn(2*RxAntNum,1)*sqrt(rho*Nv);
    s_hat=S_mmse+W*v;
    [~,Indice]=min(abs(s_hat-sym'),[],2);
    s_hat=sym(Indice);

    if ~any(all(List==s_hat,1))
        Li_current=Li_current+1;
        List=[List,s_hat];

         Ld_current=0;
        while Ld_current<Ld
            sd=S_mmse+W*diag(v)*sign(randn(2*TxAntNum,1));
            [~,Indice]=min(abs(sd-sym'),[],2);
            sd=sym(Indice);
            if ~any(all(List==sd,1))
                List=[List,sd];
                Ld_current=Ld_current+1;
            end
        end
    end
end


[~,index]=min(sum((RxSymbol-H*List).^2,1));


symest=List(:,index);


end

