function [symest] = test_bsp_b11(TxAntNum,RxAntNum,slen,RxSymbol,H,Nv,sym,delta,iterNum,TxSymbol_real)


S_mmse = (H'*H+Nv*eye(2*TxAntNum)) \ H'*RxSymbol;

gamma=-abs(S_mmse-sym');
alpha=repmat(permute(gamma,[1 3 2]),1,2*RxAntNum,1);




for iter = 1:iterNum

    ems_sym =sum(~(alpha-max(alpha,[],3)).*permute(sym,[2 3 1]),3);

    R_income_all=sum(H.*ems_sym',2);

    R_income=R_income_all-H.*ems_sym';

    choice=R_income+H.*permute(sym,[2 3 1]);

    beta=-abs(RxSymbol - choice).^2/2/Nv;

    beta=beta-beta(:,:,1);

    beta_sum = sum(beta,1);
    gamma = permute(beta_sum,[3,2,1]);

    % the passing messege without itself
    alpha_new = repmat(permute(beta_sum,[2,1,3]),1,2*RxAntNum,1)-permute(beta,[2,1,3]);

    alpha = (1-delta)*alpha_new + delta*alpha;

    symLLR = gamma;

    [~,Indice] = max(gamma);
    symest = sym(Indice);
    if symest == TxSymbol_real
        break
    end


end

end

