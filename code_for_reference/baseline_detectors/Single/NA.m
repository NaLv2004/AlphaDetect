function res = NA(RxSymbol_c,HMat,Nv,sym,norm_f,TxSymbol)
Mc=chol(inv(HMat'*HMat),"lower");
Mc=normalize(Mc,2,"norm");

RxNum=size(HMat,1);
TxNum=size(HMat,2);
beta=(TxNum/8)^(-1/3);
dqam=norm_f;
tau=1/norm(HMat'*HMat,'fro');
rho=0.9;


sym_c=reshape(sym+1j*sym',[],1);

% init=(HMat'*HMat+Nv) \ HMat'*RxSymbol_c;
init=sym_c(randperm(numel(sym_c),TxNum));


z = init;
[~,index]=min(abs(z-sym_c.'),[],2);
x=sym_c(index);

candidates=x;

for p=1:16
    z = init;
    [~,index]=min(abs(z-sym_c.'),[],2);
    x=sym_c(index);
    r=RxSymbol_c-HMat*x;
    gamma=max(dqam,norm(r)/sqrt(RxNum))*beta;

    deltaz=zeros(TxNum,1);
    for s=1:16
        z=x;

        for Ng=1:16
            p=z+rho*deltaz;
            z_old=z;

            z=p+tau*HMat'*(RxSymbol_c-HMat*p);
            deltaz=z-z_old;
        end

        zgrad=z;
        zprop=zgrad+gamma*Mc*randn(TxNum,1);

        [~,index]=min(abs(zprop-sym_c.'),[],2);
        xprop=sym_c(index);

        rprop=RxSymbol_c-HMat*xprop;

        nrprop=norm(rprop)^2;
        nr=norm(r)^2;

        alpha=min(1,exp(-nrprop+nr));
        u=rand();

        if alpha>u
            x=xprop;
            r=rprop;
            candidates=[candidates x];
        end
        gamma=max(dqam,norm(r)/sqrt(RxNum))*beta;
    end



end

dis=vecnorm(RxSymbol_c-HMat*candidates);
[~,index]=min(dis);
res=candidates(:,index);
res=[real(res);imag(res)];




end

