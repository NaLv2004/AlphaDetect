function symest = Repeat_GAI(TxAntNum,RxAntNum,slen,RxSymbol,H,Nv,sym,delta,iterNum,TxSymbol_real,norm_f)



limit=1;

% Probability for each node with slen alternative
Px = ones(2*TxAntNum,2*RxAntNum,slen)/slen;
Pxsz = size(Px);

for k = 1:iterNum

    %Means vars update with the changing Px
    Means = reshape(reshape(Px,[],Pxsz(3))*sym,Pxsz(1:2));
    Vars  = reshape(reshape(Px,[],Pxsz(3))*(sym.^2),Pxsz(1:2))-(abs(Means)).^2;


    Mu_all = sum(H.*Means',2);
    Sigma_forsum = (abs(H)).^2.*(Vars');
    Sigma_all = sum(Sigma_forsum,2);

    %Mu Sigma calculate
    Mu_temp = Mu_all-H.*Means';
    Sigma_temp = repmat(Sigma_all,1,2*TxAntNum)-Sigma_forsum+Nv;

    % Beta Update with updated Mu and Sigma
    Beta = (repmat(2*H.*(RxSymbol-Mu_temp),1,1,slen).* permute((sym-sym(1)),[3,2,1])-repmat(H.^2,1,1,slen).*permute((sym.^2-sym(1)^2),[3,2,1]))./(2*Sigma_temp);
    Beta_sum = sum(Beta,1);
    gamma = permute(Beta_sum,[3,2,1]);

    % the passing messege without itself
    Alpha = repmat(permute(Beta_sum,[2,1,3]),1,2*RxAntNum,1)-permute(Beta,[2,1,3]);

    % Px update with the alpha. here the Px update strategy is the same as the one in traditional BP algorithm
    amax_matrix = permute(max(permute(Alpha,[3,1,2])),[2,3,1]);
    a_temp = Alpha-repmat(amax_matrix,1,1,slen);
    expalpha_ij = exp(a_temp)./(repmat(sum(exp(a_temp),3),1,1,slen));

    Px = (1-delta)*expalpha_ij+delta*Px;
end


[~,Indice] = max(gamma);
symest = sym(Indice);

addloop=0;

while sum(abs(symest-TxSymbol_real))>=0.001 && addloop<=10
    addloop=addloop+1;

    Tx_add=sym(randi(slen,2*TxAntNum,1))-TxSymbol_real; 


    Rx_fake=RxSymbol+H*Tx_add;


    %test_temp=TxSymbol_real+Tx_fake;

    symest_add=RD_GAI_BP_Det(TxAntNum,RxAntNum,slen,Rx_fake,H,Nv,sym,delta,iterNum);
    symest_new=symest_add-Tx_add;

    %if sum(int8(abs(symest_new-TxSymbol_real)/norm_f))<sum(int8(abs(symest-TxSymbol_real)/norm_f))
    if sum(abs(RxSymbol-H*symest_new))<sum(abs(RxSymbol-H*symest))
        symest=symest_new;
    end

end



end
