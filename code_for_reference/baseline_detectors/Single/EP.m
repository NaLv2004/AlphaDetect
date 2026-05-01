function  symest= EP(TxAntNum,RxAntNum,slen,RxSymbol,H,Nv,sym,delta,iterNum)

Alpha = 2*ones(2*TxAntNum, 1);
Gamma = zeros(2*TxAntNum, 1);
 
Alpha_new = zeros(2*TxAntNum, 1);
Gamma_new = zeros(2*TxAntNum, 1);

h2     = zeros(2*TxAntNum, 1);            
t  = zeros(2*TxAntNum, 1);   % 腔分布的方差和均值
sigma2_p  = zeros(2*TxAntNum, 1);        
mu_p = zeros(2*TxAntNum, 1); % 替代分布的方差和均值

Sigma_q = inv(H'*H/Nv + diag(Alpha));  % 将MMSE结果作为预处理
Mu_q = Sigma_q* (H'*RxSymbol/Nv + Gamma);        



        for k = 1:iterNum      
            sig=diag(Sigma_q);
            h2=sig./(1-sig.*Alpha);
            t=h2.*(Mu_q./sig-Gamma);
            prob=exp(-(t-sym').^2./(2*h2));
            prob = prob./sum(prob,2);
            mu_p=prob*sym;
            sigma2_p=sum(prob*(sym.^2),2)-mu_p.^2;
            sigma2_p(sigma2_p<5e-7)=5e-7;

            tempAlpha=1./sigma2_p-1./ h2;
            tempGamma=mu_p./sigma2_p-t./h2;
            
            in=tempAlpha > 5e-7;
            Alpha_new(in)=tempAlpha(in);
            Gamma_new(in)=tempGamma(in);

 

        Alpha = delta* Alpha_new + (1-delta)* Alpha;
        Gamma = delta* Gamma_new + (1-delta)* Gamma;

        %EPD
        Sigma_q = inv(H'*H/Nv + diag(Alpha));
        Mu_q = Sigma_q * (H'*RxSymbol/Nv + Gamma);

        end


[~,Indice]=min(abs(Mu_q-sym'),[],2);
symest = sym(Indice);

 

end

