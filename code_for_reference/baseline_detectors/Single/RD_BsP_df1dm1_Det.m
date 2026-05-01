function [symLLR] = RD_BsP_df1dm1_Det(TxAntNum,RxAntNum,slen,RxSymbol,Hreal,Nv,sym,delta,iterNum)
%%% Simplified BsP Detection


% EMS-BP detection
symProb = zeros(2*TxAntNum, slen);
alpha = zeros(2*TxAntNum, 2*RxAntNum, slen);
beta = zeros(2*RxAntNum, 2*TxAntNum, slen);
gamma = zeros(2*TxAntNum,slen);

% intialization with LMMSE detector
K_mmse = inv(Hreal'*Hreal+Nv*eye(2*TxAntNum));
S_mmse = (Hreal'*Hreal+Nv*eye(2*TxAntNum)) \ Hreal'*RxSymbol;

for i = 1:2*TxAntNum
    symProb(i,:) = exp(-abs(sym - S_mmse(i)).^2 / (2*abs(K_mmse(i,i))));
    % 概率归一化
    symProb(i,:) = symProb(i,:)./sum(symProb(i,:));
    gamma(i,:) = log(symProb(i,:) / symProb(i,1));
    alpha(i,:,:) = repmat(gamma(i,:),2*RxAntNum,1); 
end

% EMS-BP iteration:
for iter = 1:iterNum

    % 遍历所有的factor node；
    for j = 1:2*RxAntNum

        % EMS symbol Sort
        alpha_temp = reshape(alpha(:,j,:),2*TxAntNum,slen);
        [~, idx_ems] = max(alpha_temp,[],2);
        ems_sym = sym(idx_ems);
        
        R_income_all = Hreal(j, :)*ems_sym;

        for i = 1:2*TxAntNum


            % 首先计算所有Edge对应的Hs；% 注意千万不能用共轭转置
            R_income = R_income_all - Hreal(j, i)*ems_sym(i);
            
            % 开始计算每个outcoming edge的beta消息
            x_upp = Hreal(j,i)*sym(1) + R_income;
            y_upp = Hreal(j,i).*sym(2:slen) + R_income; 
            x_temp = -abs(RxSymbol(j) - x_upp).^2/2/Nv;
            y_temp = -abs(RxSymbol(j) - y_upp).^2/2/Nv;
            beta(j,i,2:slen) = y_temp - x_temp;
        end 
    end

    % 更新alpha消息

    beta_sum = sum(beta,1);
    gamma = permute(beta_sum,[3,2,1]);

    % the passing messege without itself
    alpha_new = repmat(permute(beta_sum,[2,1,3]),1,2*RxAntNum,1)-permute(beta,[2,1,3]);	

    alpha = (1-delta)*alpha_new + delta*alpha;

end
symLLR = gamma;
        
end