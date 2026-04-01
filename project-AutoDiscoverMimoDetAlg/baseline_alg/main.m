% BP Algorithm with GAI test in LLR domain
clear;
clc;

rand('state',114514);
randn('state',114514);

% profile on;
EbN0db      =6:2:14;
SampleNum   = 100000;       % max sample no.aaa
delta       = 0.7; %0.55;          % damping factor
isCorr = 0;
rho = 0.2;  % related channel parameter
max_frame   =50;
ModType 	= 4;

kbestCount=0;


% Detection Parameter Set
TxAntNum    =16;
RxAntNum    =48;
iterNum     =7;% 7;        % iter no.
N 			= 1; 						% Block length
InfoLen 	= N * ModType * TxAntNum; 	% Information length

switch(ModType)
    case 2
        norm_f=1/sqrt(2);
        QAM=4;
        s=[-1 +1]';
        sym = s*norm_f; % normalized symbol set
        %cons = getConstellation(ModType);
    case 4
        norm_f = 1/sqrt(10);
        QAM=16;
        s=[-1 -3 +1 +3]';
        sym = s*norm_f; % normalized symbol set
        %cons = getConstellation(ModType);
    case 6
        norm_f=1/sqrt(42);
        QAM=64;
        s=[-3 -1 -5 -7 3 1 5 7 ]';
        sym = s*norm_f; % normalized symbol set
        %cons = getConstellation(ModType);
    case 8
        QAM=256;
        norm_f=1/sqrt(170);
        s=[-5 -7 -3 -1 -11 -9 -13 -15 5 7 3 1 11 9 13 15]';
        sym = s*norm_f; % normalized symbol set
        %cons = getConstellation(ModType);
end

slen = length(sym);
H           = zeros(2*RxAntNum,2*TxAntNum);
RxSymbol    = zeros(2*RxAntNum,1); % Received


tic
for nEN =1:length(EbN0db)

    % initialization
    SNR = EbN0db(nEN);
    error_frame = 0;
    error_bits = 0;


    TxBitsLLR = zeros(InfoLen, 1);

    loop = 0;
    while error_frame < max_frame && loop < SampleNum

        loop = loop + 1;

        TxBits = randi([0, 1], [InfoLen, 1]);

        TxSymbol = GrayMapQAM(TxBits', ModType).';
        % TxSymbol=qammod(TxBits,QAM,"gray",'UnitAveragePower',true,'InputType','bit');
        TxSymbol_real=[real(TxSymbol); imag(TxSymbol)];
        % Channel Matrix
        if isCorr
            HMat = channel_mimo(RxAntNum,TxAntNum,rho); %Correlated channel
        else
            HMat = sqrt(0.5) *(randn(RxAntNum, TxAntNum) + sqrt(-1)*randn(RxAntNum, TxAntNum)); %Rayleigh channel
        end




        % Received and sigma^2
        [RxSymbol_c, Nv] = acwgn_EbN0(HMat*TxSymbol, 1, SNR, TxAntNum, RxAntNum, ModType, 1);	% Add Gaussian noise

        if (mod(loop,10) == 0)
            fprintf("\nNow Iter: %d\tNow SNR: %d\tNow FER: %fNow BER: %f\tNow Error Frame: %d \tNow Error Bits: %d", ...
                loop, EbN0db(nEN), error_frame/loop, error_bits/loop/TxAntNum/ModType, error_frame, error_bits);
        end

        % Complex channel matrix to real
        Hreal = [real(HMat), -imag(HMat); imag(HMat), real(HMat)];

        RxSymbol(1:RxAntNum,1)=real(RxSymbol_c);
        RxSymbol(RxAntNum+1:end,1)=imag(RxSymbol_c);

%         Hreal = load("C:\Users\许嘉研\Downloads\Hreal0128.txt");
%         RxSymbol = load("C:\Users\许嘉研\Downloads\RxSig0128.txt");
%         RxSymbol = RxSymbol';
%         Nv = 1.01335168;

        %symLLR = Bsp_b11_fast(TxAntNum,RxAntNum,slen,RxSymbol,Hreal,Nv,sym,delta,iterNum);[~,Indice] = max(symLLR);symest = sym(Indice);

        %symest = RD_GAI_BP_Det(TxAntNum,RxAntNum,slen,RxSymbol,Hreal,Nv,sym,delta,iterNum);

        %symest = AMP(TxAntNum,RxAntNum,slen,RxSymbol,Hreal,Nv,sym,delta,iterNum);
        symest= EP(TxAntNum,RxAntNum,slen,RxSymbol,Hreal,Nv,sym,delta,iterNum);
        %[symLLR, ~] = Bsp_dm_df(TxAntNum,RxAntNum,slen,RxSymbol,Hreal,Nv,sym,delta,iterNum,1,1);

        % [symLLR] = Bsp_b11_fast(TxAntNum,RxAntNum,slen,RxSymbol,Hreal,Nv,sym,delta,iterNum);
        % %symLLR = squeeze(symLLR).';
        % [~, Indice] = max(symLLR, [], 1); % 寻找每一列中概率最大的索引
        % symest = sym(Indice);       % 将索引映射为符号值，并转置为列向量

        %symest = EP(TxAntNum,RxAntNum,slen,RxSymbol,H,Nv,sym,delta,iterNum);
        %symest= AMP(TxAntNum,RxAntNum,slen,RxSymbol,H,Nv,sym,delta,iterNum);
        %symest = MMSE(TxAntNum,RxAntNum,slen,RxSymbol,Hreal,Nv,sym,delta,iterNum);
        % symest =NA(RxSymbol_c,HMat,Nv,sym,norm_f,TxSymbol);
        % symest = GD(TxAntNum,RxAntNum,RxSymbol,Hreal,sym,iterNum,Nv);

        %symest = MDPI_GAI(TxAntNum,RxAntNum,slen,RxSymbol,Hreal,Nv,sym,delta,iterNum);
        % symest = test_GAI(TxAntNum,RxAntNum,slen,RxSymbol,Hreal,Nv,sym,delta,iterNum,TxSymbol_real);
        %symest = AltMin(TxAntNum,RxAntNum,slen,RxSymbol,Hreal,Nv,sym,delta,iterNum,TxSymbol_real);
       


        % symest = FC_GAI_BP_Det(TxAntNum,RxAntNum,slen,RxSymbol,Hreal,Nv,sym,delta,iterNum);

        %symest =  Repeat_GAI(TxAntNum,RxAntNum,slen,RxSymbol,Hreal,Nv,sym,delta,iterNum,TxSymbol_real,norm_f);
        %symest =  Repeat_GAI_IDD(TxAntNum,RxAntNum,slen,RxSymbol,Hreal,Nv,sym,delta,iterNum,TxSymbol_real,norm_f);


        % no use
        % symest =  Repeat_Bsp(TxAntNum,RxAntNum,slen,RxSymbol,Hreal,Nv,sym,delta,iterNum,TxSymbol_real,norm_f);
        % symLLR = RD_BsP_df1dm1_Det(TxAntNum,RxAntNum,slen,RxSymbol,Hreal,Nv,sym,delta,iterNum);[~,Indice] = max(symLLR);symest = sym(Indice);

        % DANGER!DANGER!
        %         symest=  maxlikehood(RxSymbol,Hreal,sym);


        % QR for K_BEST
        %RxSymbol = load("C:\信道编码\2D\kbest_test\rx_qr.txt");
        %Hreal = load("C:\信道编码\2D\kbest_test\h_real_qr.txt");
%         [bQ, bR] = qr(Hreal);
%         R = bR(1:2*TxAntNum, :);
%         Q = bQ(:, 1:2*TxAntNum);
%         z = Q' * RxSymbol;
% 
%                  %[symest,kbesttime,PEDtemp] =mcts_det(R, sym, z,Nv,TxSymbol_real); kbestCount=kbestCount+kbesttime;averagekbest=kbestCount/loop;PEDCount=[PEDCount;PEDtemp];
%         symest=K_Best(R, sym', z,128);

        % 用那个祖传格雷码的时候用这4行
        [~,indiceest]=min(abs(sym'-symest),[],2);
        TxBits_est=de2bi(indiceest-1,log2(size(sym,1)),'left-msb');
        TxBits_est=[TxBits_est(1:TxAntNum,:) TxBits_est(TxAntNum+1:end,:)]';
        TxBits_est=TxBits_est(:);


        % RxSymbol_est=symest(1:TxAntNum)+1j*symest(TxAntNum+1:end);
        % TxBits_est=qamdemod(RxSymbol_est,QAM,"gray","OutputType","bit","UnitAveragePower",true);
        % RxSymbol_est=symest(1:TxAntNum)+1j*symest(TxAntNum+1:end);
        % TxBits_est=qamdemod(RxSymbol_est,QAM,"gray","OutputType","bit","UnitAveragePower",true);

        err=nnz(TxBits_est-TxBits);
        error_bits = error_bits + err;
        error_frame = error_frame + ~~err;


    end

    BER(nEN) = error_bits / TxAntNum / ModType / loop;
    FER(nEN) = error_frame / loop;

end
toc

