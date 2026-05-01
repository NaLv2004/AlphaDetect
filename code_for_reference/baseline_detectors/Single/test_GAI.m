 function symest= test_GAI(TxAntNum,RxAntNum,slen,RxSymbol,H,Nv,sym,delta,iterNum,TxSymbol_real)


symp=permute(sym,[2 3 1]);

% Nestrerov init
lambda=1/norm(H'*H,'fro');
% lambda=1./diag(H'*H) ;
beta=0.9;
Tx=zeros(2*TxAntNum,1);
symest=zeros(2*TxAntNum,1);

A=eye(2*TxAntNum)-lambda.*(H'*H);
I=eye(2*TxAntNum);
CC=lambda.*H'*RxSymbol;
for loop=1:3
    d=zeros(2*TxAntNum,1);
    for i=1:10
        %% original
        % Tx_new=Tx-beta*d;
        % grad= H'*(H*Tx_new-RxSymbol);
        % d=beta*d+lambda*grad;
        % Tx=Tx- d;

        %% 一阶
        d=Tx-A*(Tx-beta*d)-CC;
        Tx=Tx-d;
    end
    
    % Tx=(I + A + A^2 + A^3 + A^7 *(1 + beta)^7 - A^6* (1 + beta)^5* (-1 + 5 *beta) + A^5 *(1 + beta)^3 *(1 - 3* beta + 6* beta^2) - A^4 *(-1 + beta^4))* CC + A^4* (beta^4*I + A^4* (1 + beta)^7 + 5*A^2 *beta^2 *(1 + beta)^3 *(3 + beta) - A^3 *beta* (1 + beta)^5 *(7 + beta) - 2* A* beta^3* (5 + 8* beta + 3 *beta^2))* Tx;
    
    Muzji= H*Tx-H.*Tx';
    Beta=(2*H.*(RxSymbol-Muzji).*(symp-sym(1))-H.^2.*(symp.^2-sym(1)^2));
    
    
    Beta_sum = sum(Beta,1);
    % the passing messege without itself
    alpha  =  Beta_sum-Beta ;
    
    
    for k = 1:3
        
        % ---------------------------
        % max
        [~,index]=max(alpha,[],3);
        Muz=H.*sym(index);
        
        % mean
        % Px=exp(alpha-max(alpha,[],3));
        % Px=Px./sum(Px,3);
        % Muz=H.*sum(Px.*symp,3);
        % -----------------------------
        Muzji=sum(Muz,2)-Muz;
        
        % ---------------------------
        % without var
        Beta=(2*H.*(RxSymbol-Muzji).*(symp-sym(1))-H.^2.*(symp.^2-sym(1)^2))./(2*Nv);
        
        % with var
        % sigmaz2=H.^2.*(sum(Px.*symp.^2,3)-sum(Px.*symp,3).^2);
        % sigmaz2ji=sum(sigmaz2,2)-sigmaz2+Nv;
        % Beta=(2*H.*(RxSymbol-Muzji).*(symp-sym(1))-H.^2.*(symp.^2-sym(1)^2))./(2*sigmaz2ji);
        % ---------------------------
        
        
        Beta_sum = sum(Beta,1);
        % the passing messege without itself
        alpha  =  Beta_sum-Beta ;
        
    end
    gamma = squeeze(Beta_sum);
    
    % ----------------------
    % Tx from mean
    % P=exp(gamma-max(gamma,[],2));
    % P=P./sum(P,2);
    % Tx=P*sym;
    % [~,Indice] = max(gamma,[],2);
    % symest = sym(Indice);
    
    % Tx from max
    [~,indice ]=max(gamma,[],2);
    Tx=sym(indice);
    
    if Tx==symest
        break;
    else
        symest=Tx;
    end
    % ----------------------
end




