%trapping set统计特性
clear all；
clc

SampleNum   = 20000;       % max sample no.
iterNum     = 40;% 7;        % iter no.   
delta       = 0.5;          % damping factor
rho = 0.3;
max_nberr   = 20;

TxAntNum    = 16;
RxAntNum    = 32;
ModType 	= 4; 					

switch(ModType)
    case 2
        norm_f=1/sqrt(2);
        QAM=4;
        s=[ -1 +1 ]';
    case 4
        norm_f = 1/sqrt(10);
        QAM=16;
        s=[ -3 -1 +1 +3  ]';
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

EbN0db      = 16:2:18;
SampleTest  = zeros(length(EbN0db),1);  

H           = zeros(2*RxAntNum,2*TxAntNum);  
RxSymbol    = zeros(2*RxAntNum,1); % Received
TxSym_Detected = zeros(2*TxAntNum, 1);

slen = length(s);
sym = s*norm_f; % normalized symbol set

error_real = zeros(1,length(EbN0db));
error_sym = zeros(iterNum, length(EbN0db));
error_MMSE = zeros(1, length(EbN0db));
error_s_MMSE = zeros(1, length(EbN0db));

for nEN =1:length(EbN0db)
    
    % initialization 
    SNR = EbN0db(nEN)
    
    loop = 0;
    while error_real(nEN) < max_nberr
    %while SampleTest < SampleNum
        loop = loop + 1;
        % Data send via the channels
%         TxBits = zeros(InfoLen,1);
        TxBits = randi([0, 1], [InfoLen, 1]);
        Tx_mat = reshape(TxBits,ModType,InfoLen/ModType);
        Tx_H   = bi2de(Tx_mat.','left-msb');
        TxSymbol = GrayMapQAM((TxBits).', ModType).';
        
        % Channel Matrix
        HMat = sqrt(0.5) *(randn(RxAntNum, TxAntNum) + sqrt(-1)*randn(RxAntNum, TxAntNum));
        %HMat = TheOne;
        % Complex channel matrix to real
        H = [real(HMat), -imag(HMat); imag(HMat), real(HMat)];
        
        % Received and sigma^2
        [RxSymbol_c, Nv] = acwgn_EbN0(HMat*TxSymbol, 1, SNR, TxAntNum, RxAntNum, ModType, 1);	% Add Gaussian noise
        %RxSymbol_c = HMat*TxSymbol;
        %Nv = 0;
        Nv_MMSE = Nv;
        RxSymbol(1:RxAntNum,1)=real(RxSymbol_c);
        RxSymbol(RxAntNum+1:end,1)=imag(RxSymbol_c);

        % Complex to real for transmitted signal
        TxSymbol_r_0 = zeros(2*TxAntNum, 1);
        TxSymbol_r_0(1:TxAntNum)=1/norm_f*real(TxSymbol);
        TxSymbol_r_0(TxAntNum+1:end)=1/norm_f*imag(TxSymbol);
        Tx_B = symboltobits(TxSymbol_r_0, 2*TxAntNum, QAM); 
        TxSymbol_r_0 = round(TxSymbol_r_0);

        TxSym_Detected = Repeat_GAI(TxAntNum,RxAntNum,slen,RxSymbol,H,Nv,sym,delta,iterNum,TxSymbol_r_0,norm_f); 
        TxSym_Detected = round(TxSym_Detected/norm_f);
            
        Rx_B_bp = symboltobits(TxSym_Detected,2*TxAntNum,QAM); 
        Rx_S_bp = TxSym_Detected(1:TxAntNum)*norm_f+TxSym_Detected(TxAntNum+1:end)*norm_f*1j;
        error_real(nEN) = error_real(nEN)+nnz(Rx_B_bp-Tx_B);

        end

    SampleTest(nEN) = loop; %SampleNum;
end

%error_sym = error_sym ./ repmat((TxAntNum .* SampleTest'),iterNum,1);
error_real = error_real ./ (TxAntNum * ModType.* SampleTest');

semilogy(EbN0db, error_real, 'cv-','LineWidth',1.25);
grid on;
legend('ABP-Repeat, iter=40'); set(gca,'FontSize',8);
xlabel('Average Received SNR(dB)')
ylabel('BER')

 %figure('color','w');
 %semilogy(EbN0db, ave_wrong_si, 'bo-','LineWidth',2);
 %grid on;
 %legend('GAI-BP,TxAntNum=RxAntNum=32,iter=40');
 %xlabel('Average Received SNR(dB)')
 %ylabel('Average wrongly detected si number');  
 
 %figure('color','w');
 %semilogy(EbN0db, ynum_ratio, 'm>-','LineWidth',2);
 %grid on;
 %legend('GAI-BP,TxAntNum=RxAntNum=32,iter=40');
 %xlabel('Average Received SNR(dB)')
 %ylabel('trapping set ratio');  
 
 %figure('color','w');
 %semilogy(EbN0db, ave_ynum, 'r>-','LineWidth',2);
 %grid on;
 %legend('GAI-BP,TxAntNum=RxAntNum=32,iter=40');
 %xlabel('Average Received SNR(dB)')
 %ylabel('average number of wrong obeservation nodes');  
