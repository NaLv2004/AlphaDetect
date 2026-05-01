function symest = Repeat_GAI_IDD(TxAntNum,RxAntNum,slen,RxSymbol,H,Nv,sym,delta,iterNum,TxSymbol_real,norm_f)



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



if sum(abs(RxSymbol-H*symest))>3*Nv 
%if sum(abs(symest-TxSymbol_real))~=0
 

    Tx_add=-sign(symest)*norm_f*2; 
    %Tx_add=-sign(TxSymbol_real)*norm_f*2; 
    %Tx_add=sym(randi(slen,2*TxAntNum,1))-TxSymbol_real;


    Rx_fake=RxSymbol+H*Tx_add;

    Px = ones(2*TxAntNum,2*RxAntNum,slen)/slen;
    Px_fake = ones(2*TxAntNum,2*RxAntNum,slen)/slen;


    for k = 1:iterNum


         %Means vars update with the changing Px
        Means_fake = reshape(reshape(Px_fake,[],Pxsz(3))*sym,Pxsz(1:2));
        Vars_fake  = reshape(reshape(Px_fake,[],Pxsz(3))*(sym.^2),Pxsz(1:2))-(abs(Means_fake)).^2;
    
    
        Mu_all_fake = sum(H.*Means_fake',2);
        Sigma_forsum_fake = (abs(H)).^2.*(Vars_fake');
        Sigma_all_fake = sum(Sigma_forsum_fake,2);
    
        %Mu Sigma calculate
        Mu_temp_fake = Mu_all_fake-H.*Means_fake';
        Sigma_temp_fake = repmat(Sigma_all_fake,1,2*TxAntNum)-Sigma_forsum_fake+Nv;
    
        % Beta Update with updated Mu and Sigma
        Beta_fake = (repmat(2*H.*(Rx_fake-Mu_temp_fake),1,1,slen).* permute((sym-sym(1)),[3,2,1])-repmat(H.^2,1,1,slen).*permute((sym.^2-sym(1)^2),[3,2,1]))./(2*Sigma_temp_fake);
        Beta_sum_fake = sum(Beta_fake,1);
    
        Alpha_fake = repmat(permute(Beta_sum_fake,[2,1,3]),1,2*RxAntNum,1)-permute(Beta_fake,[2,1,3]);


        Alpha_add2=zeros(2*TxAntNum,2*RxAntNum,slen);

        for t=1:2*TxAntNum
            offset=-int8(Tx_add(t)/norm_f);
            sym_forpa=int8(sym/norm_f);

            temp=zeros(2*RxAntNum,slen);

            for s=1:slen
                if ismember(sym_forpa(s)+offset,sym_forpa)
                    in2=sym_forpa==sym_forpa(s)+offset;
                    temp(:,in2)=Alpha_fake(t,:,s);
                end
            end
            Alpha_add2(t,:,:)=temp;
        end
        Alpha=Alpha+Alpha_add2;

        amax_matrix = permute(max(permute(Alpha,[3,1,2])),[2,3,1]);
        a_temp = Alpha-repmat(amax_matrix,1,1,slen);
        expalpha_ij = exp(a_temp)./(repmat(sum(exp(a_temp),3),1,1,slen));

        Px = (1-delta)*expalpha_ij+delta*Px;


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
    
        Alpha = repmat(permute(Beta_sum,[2,1,3]),1,2*RxAntNum,1)-permute(Beta,[2,1,3]);
        Alpha_add1=zeros(2*TxAntNum,2*RxAntNum,slen);

        for t=1:2*TxAntNum
            offset=int8(Tx_add(t)/norm_f);
            sym_forpa=int8(sym/norm_f);

            temp=zeros(2*RxAntNum,slen);

            for s=1:slen
                if ismember(sym_forpa(s)+offset,sym_forpa)
                    in2=sym_forpa==sym_forpa(s)+offset;
                    temp(:,in2)=Alpha(t,:,s);
                end
            end
            Alpha_add1(t,:,:)=temp;
        end
     
       
 

        
        Alpha_fake=Alpha_fake+Alpha_add1;

        


        amax_matrix_fake = permute(max(permute(Alpha_fake,[3,1,2])),[2,3,1]);
        a_temp_fake = Alpha_fake-repmat(amax_matrix_fake,1,1,slen);
        expalpha_ij_fake = exp(a_temp_fake)./(repmat(sum(exp(a_temp_fake),3),1,1,slen));

        Px_fake = (1-delta)*expalpha_ij_fake+delta*Px_fake;


    end
    gamma = permute(Beta_sum,[3,2,1]);


    [~,Indice] = max(gamma);
    symest = sym(Indice);

   


end



end
