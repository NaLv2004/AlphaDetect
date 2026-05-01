% BP Algorithm with GAI test in LLR domain
clear all;
clc
rand('state',789);
randn('state',789);
warning('off','comm:obsolete:randint');

% profile on;
SampleNum   = 100000;       % max sample no.
iterNum     = 5;% 7;        % iter no.   
delta       = 0; %0.55;          % damping factor
rho = 0.3;
max_error_frame = 300;

TxAntNum    = 16;
RxAntNum    = 32;
ModType 	= 4; 		
isCorr = 0;
% norm_f      = 1 / sqrt(10);  % normalize the Transimite Symnbol 

switch(ModType)
    case 2
        norm_f=1/sqrt(2);
        QAM=4;
        s=[-1 +1]';
    case 4
        norm_f = 1/sqrt(10);
        QAM=16;
        s=[-3 -1 +1 +3]';
    case 6
        norm_f=1/sqrt(42);
        QAM=64;
        s=[-7  -5 -3 -1 +1 +3 +5 +7 ]';
    case 8
        QAM=256;
        norm_f=1/sqrt(170);
        s=[-15 -13 -11 -9 -7  -5 -3 -1 +1 +3 +5 +7 +9 +11 +13 +15]';
end
N 			= 1; 						% Block length
InfoLen 	= N * ModType * TxAntNum; 	% Information length

EbN0db      = 10:2:16;
SampleTest  = zeros(length(EbN0db),1);  

H           = zeros(2*RxAntNum,2*TxAntNum);  
RxSymbol    = zeros(2*RxAntNum,1); % Received
TxSym_Detected = zeros(2*TxAntNum, 1);

slen = length(s);
sym = s*norm_f; % normalized symbol set

tic
for nEN =1:length(EbN0db)
    
    % initialization 
    SNR = EbN0db(nEN);
    loop = 0;
    error_frame = 0;
    error_num = 0;


    while error_frame < max_error_frame && loop < SampleNum
        
        loop = loop + 1;
        
        if (mod(loop,10) == 0)
            fprintf("\nNow Iter: %d\tNow SNR: %d\tNow Error Frame: %d\tNow Error Bits: %d", loop, EbN0db(nEN), error_frame, error_num);
        end
        
        TxBits = randi([0, 1], [InfoLen, 1]);
        Tx_mat = reshape(TxBits,ModType,InfoLen/ModType);
        Tx_H   = bi2de(Tx_mat.','left-msb');
        TxSymbol = GrayMapQAM((TxBits).', ModType).';
        
        % Channel Matrix
        if isCorr
            HMat = channel_mimo(RxAntNum,TxAntNum,rho); %Correlated channel
        else
            HMat = sqrt(0.5) *(randn(RxAntNum, TxAntNum) + sqrt(-1)*randn(RxAntNum, TxAntNum)); %Rayleigh channel
        end
        
        % Complex channel matrix to real
        H = [real(HMat), -imag(HMat); imag(HMat), real(HMat)];
        
        % Received and sigma^2
        [RxSymbol_c, Nv] = acwgn_EbN0(HMat*TxSymbol, 1, SNR, TxAntNum, RxAntNum, ModType, 1);	% Add Gaussian noise
        Nv_MMSE = Nv;

        RxSymbol(1:RxAntNum,1)=real(RxSymbol_c);
        RxSymbol(RxAntNum+1:end,1)=imag(RxSymbol_c);

        % Complex to real for transmitted signal
        TxSymbol_r_0 = zeros(2*TxAntNum, 1);
        TxSymbol_r_0(1:TxAntNum)=1/norm_f*real(TxSymbol);
        TxSymbol_r_0(TxAntNum+1:end)=1/norm_f*imag(TxSymbol);
        Tx_B = symboltobits(TxSymbol_r_0, 2*TxAntNum, QAM); 

        % EP 14TCOM
         delta=0.5; 
% 
%         % EP检测器
% 
%         % EP init
        Alpha = 2*ones(2*TxAntNum, 1);     Gamma = zeros(2*TxAntNum, 1);   sym = s*norm_f;
        Alpha_new = zeros(2*TxAntNum, 1);  Gamma_new = zeros(2*TxAntNum, 1);

        h2     = zeros(2*TxAntNum, 1);            t  = zeros(2*TxAntNum, 1);   % 腔分布的方差和均值
        sigma2_p  = zeros(2*TxAntNum, 1);         mu_p = zeros(2*TxAntNum, 1); % 替代分布的方差和均值

        % Preprocessing
        Sigma_q = (H'*H/Nv + diag(Alpha))\ eye(2*TxAntNum);  % 将MMSE结果作为预处理
        Mu_q = Sigma_q* (H'*RxSymbol/Nv + Gamma);          % 

        for k = 1:iterNum      
            for i = 1:2*TxAntNum        
                h2(i) = Sigma_q(i,i)/(1-Sigma_q(i,i)*Alpha(i));     % 腔边缘分布的方差
                t(i)  = h2(i) * (Mu_q(i)/Sigma_q(i,i)-Gamma(i));   % 腔边缘分布的期望

                logprob = -(sym - t(i)).^2/(2*h2(i));      % 计算每个星座点的后验概率
        %         maxlogprob = max(logprob);                 % 找出最大的后验概率
        %         unitmaxprob = logprob - maxlogprob;        % 归一化后验概率
        %         prob = exp(unitmaxprob);                   % 归一化后的后验概率
                prob = exp(logprob);

                prob = prob/sum(prob);
                mu_p(i) = sum(sym.*prob);                  % 计算替代分布的均值
                temp_sigma2_p = sum((sym.^2).*prob) - (mu_p(i))^2; % 替代分布的方差
                sigma2_p(i) = max( [temp_sigma2_p 5e-7]);

                tempAlpha = 1/sigma2_p(i) - 1/ h2(i); 
                tempGamma = mu_p(i)/sigma2_p(i) - t(i)/h2(i);

                if tempAlpha > 5e-7
                    Alpha_new(i) = tempAlpha;
                    Gamma_new(i) = tempGamma;
                end
            end

        Alpha = delta* Alpha_new + (1-delta)* Alpha;
        Gamma = delta* Gamma_new + (1-delta)* Gamma;

        %EPD
        Sigma_q = (H'*H/Nv + diag(Alpha))\eye(2*TxAntNum);
        Mu_q = Sigma_q * (H'*RxSymbol/Nv + Gamma);

        end
        
        TxSym_Detected = HD_EP_all(Mu_q/norm_f,s); % hard decision
        Rx_B_bp = symboltobits(TxSym_Detected, 2*TxAntNum,QAM); 
        error_num = error_num + nnz(Rx_B_bp-Tx_B);
        
        if nnz(Rx_B_bp-Tx_B)
            
            error_frame = error_frame + 1;
            
        end
    
    end
    
    
    FER(nEN) = error_frame / loop;
    BER(nEN) = error_num / (TxAntNum * ModType * loop);
    SampleTest(nEN) = loop; %SampleNum;
end
toc


