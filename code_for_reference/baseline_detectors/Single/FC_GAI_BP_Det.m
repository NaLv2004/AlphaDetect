function symest = FC_GAI_BP_Det(TxAntNum,RxAntNum,slen,RxSymbol,H,Nv,sym,delta,iterNum)
symp=permute(sym,[2 3 1]);
xest=(H'*H+Nv*eye(2*TxAntNum)/2)\H'*RxSymbol;
temp=xest-sym';
Px=repmat(permute(~(temp-min(temp,2)),[1 3 2]),1,2*RxAntNum,1);

for k = 1:iterNum

    Muz=H.*sum(Px.*symp,3)';
    sigmaz2=H.^2.*(sum(Px.*symp.^2,3)-sum(Px.*symp,3).^2)';

    Muzji=sum(Muz,2)-Muz;
    sigmaz2ji=sum(sigmaz2,2)-sigmaz2+Nv;

    Beta=(2*H.*(RxSymbol-Muzji).*(symp-sym(1))-H.^2.*(symp.^2-sym(1)^2))./(2*sigmaz2ji);

    Alpha = permute(sum(Beta,1)-Beta,[2 1 3]);

    Px_new=exp(Alpha-max(Alpha,[],3));
    Px_new=Px_new./sum(Px_new,3);
    Px = (1-delta)*Px_new+delta*Px;

end

gamma = squeeze(sum(Beta,1));

[~,Indice] = max(gamma,[],2);
symest = sym(Indice);

