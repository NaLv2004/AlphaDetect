clc; 
clear;

nchoosek(16,4)

Nt = 32; % n of transmitting antennas
Nr = 64; % n of receiving antennas
CodeLen = 1024;
EbN0_min = 4.5;
EbN0_max = 5.5;
EbN0_step = 0.5;
ModType = 2;
% CodeRate = 676 ./ 1024;
CodeRate = 26.*20 ./ CodeLen;
EbN0_len = round((EbN0_max - EbN0_min) /EbN0_step) + 1;
fer = zeros(EbN0_len ,1);
fer_sim_qfunc = zeros(EbN0_len, 1);
fer_sim_qfunc_irregular = zeros(EbN0_len, 1);
n_sim_channels = 100;
n_sim_qfunc = 100;
w_min = 16;    % minimum weight
% A_w_min = 1240 .* 1240;    % n of minimum weight codewords
% A_w_min = 728 * 120;    % 25*20
% A_w_min = 728 * 216;    % 25*22
% A_w_min = 728 .* 472;   % 25*24
% A_w_min = 728 .* 152;   % 25*21
A_w_min = 1240 .* 120; % 26*20
atten = 2;
S_square_stored = zeros(n_sim_channels *Nt* 2, 1);
S_square_bad_stored = zeros(n_sim_channels *Nt* 2, 1);
S_square_good_stored = zeros(n_sim_channels *Nt* 2, 1);
n_bad_channels = 15; % subchannel 1~15 are interleaved together, 16~32 are interleaved together
n_good_channels = Nt - n_bad_channels;
N_info_irregular = n_bad_channels .* 16 + (26-n_bad_channels) .* 26;
CodeRate_irregular = N_info_irregular ./ CodeLen;


%% code weight profile
w_min_lr = 32;
w_min_hr = 16;
A_w_min_lr = 48 .* 620;
A_w_min_hr = 81 .* 1240;

%% power allocation
mimo_rate_matching_enlarge_ratio = 1.85;
sum_tmp = n_bad_channels * mimo_rate_matching_enlarge_ratio + (Nt - n_bad_channels) * 1.0;
power_bad_channel = mimo_rate_matching_enlarge_ratio ./ sum_tmp .* Nt;
power_good_channel = 1.0 ./ sum_tmp .* Nt;

iEbN0 = 1;
for nEbN0 = EbN0_min : EbN0_step : EbN0_max
    if (atten == 2)
        tmp = 10.^(nEbN0 ./ 10) * ModType * CodeRate * Nt;
        % tmp = pow(10, nSNR / 10) * symbol_length  * Nt;
        sigma_sq = (Nt * Nr) / tmp;
    else 
        tmp = 10.^(nEbN0 ./ 10);
        sigma_sq = 1 ./ tmp ./ 2./ CodeRate;
    end
    qfunc_avg = 0.0;
    qfunc_sum = 0.0;
    qfunc_sum_irregular_lr = 0.0;
    qfunc_avg_irregular_lr = 0.0;
    qfunc_sum_irregular_hr = 0.0;
    qfunc_avg_irregular_hr = 0.0;
    singular_values_sum = 0.0;
    for i_sim_channel = 1:n_sim_channels
        H = randn(Nr, Nt).*sqrt(0.5)+1i*randn(Nr, Nt).*sqrt(0.5);
        % [U,S,V] = svd(H);
        S = svd(H);
        S_square = S .^ 2;
        S_square_stored((i_sim_channel-1)*Nt+1 : i_sim_channel*Nt) = S_square;
        S_square_bad_stored((i_sim_channel-1)*n_bad_channels+1 : i_sim_channel*n_bad_channels) = S_square(Nt-n_bad_channels+1 : Nt);
        S_square_good_stored((i_sim_channel-1)*n_good_channels+1 : i_sim_channel*n_good_channels) = S_square(1:n_good_channels);
        singular_values_sum = singular_values_sum + sum(S_square, 'all');
        % qfunc_tmp = qfunc(sqrt(w_min .* tmp .* CodeRate .* 2));
        % qfunc(sqrt( w_min .* h2_avg ./ (sigma_sq.^2)));
        % qfunc_sum = qfunc_sum + qfunc_tmp;
    end
    for i_sim_qfunc_regular = 1 : n_sim_qfunc
        w_min_half = round(w_min ./ 2);
        selected_pos = randi([1,n_sim_channels*Nt],w_min,1);
        selected_singular_square = S_square_stored(selected_pos);
        sum_tmp = sum(selected_singular_square);
        qfunc_tmp = qfunc(sqrt(sum_tmp./ (sigma_sq)));
        % qfunc_tmp
        qfunc_sum = qfunc_sum + qfunc_tmp;
    end
    tmp = 10.^(nEbN0 ./ 10) * ModType * CodeRate_irregular * Nt;
        % tmp = pow(10, nSNR / 10) * symbol_length  * Nt;
    sigma_sq = (Nt * Nr) / tmp;
    for i_sim_qfunc_irregular = 1 : n_sim_qfunc
        % w_min_half = round(w_min ./ 2);
        selected_pos = randi([1,n_bad_channels*Nt],w_min_lr,1);
        selected_singular_square = S_square_bad_stored(selected_pos);
        sum_tmp = sum(selected_singular_square).*power_bad_channel;
        qfunc_tmp = qfunc(sqrt(sum_tmp./ (sigma_sq)));
        % qfunc_tmp
        qfunc_sum_irregular_lr = qfunc_sum_irregular_lr + qfunc_tmp;

        selected_pos = randi([1,n_good_channels*Nt],w_min_hr,1);
        selected_singular_square = S_square_good_stored(selected_pos);
        sum_tmp = sum(selected_singular_square).*power_good_channel;
        qfunc_tmp = qfunc(sqrt(sum_tmp./ (sigma_sq)));
        % qfunc_tmp
        qfunc_sum_irregular_hr = qfunc_sum_irregular_hr + qfunc_tmp;
    end
    singular_values_avg = singular_values_sum ./ (Nt .* n_sim_channels);
    h2_avg = singular_values_avg;
    qfunc_avg = qfunc_sum ./ n_sim_qfunc;
    qfunc_avg_irregular_lr = qfunc_sum_irregular_lr ./ n_sim_qfunc;
    qfunc_avg_irregular_hr = qfunc_sum_irregular_hr ./ n_sim_qfunc;
    fprintf("EbN0: %f\n qfunc_avg_lr: %.20f\n qfunc_avg_hr: %.20f\n",nEbN0, qfunc_avg_irregular_lr, qfunc_avg_irregular_hr);
    if (atten == 1)
        % sqrt(2 .* w_min .* 1./ sigma_sq)
        % qfunc(sqrt(w_min .* 1./ sigma_sq))
        % fer (iEbN0, 1) = A_w_min .* qfunc(sqrt(w_min .* 1./ sigma_sq));
        fer (iEbN0, 1) = A_w_min .* qfunc(sqrt(w_min .* tmp .* CodeRate .* 2));
    else
        fer (iEbN0, 1) = A_w_min .* qfunc(sqrt( w_min .* h2_avg ./ (sigma_sq)));
        fer_sim_qfunc (iEbN0, 1) = A_w_min .* qfunc_avg;
        fer_sim_qfunc_irregular (iEbN0, 1) = A_w_min_hr .* qfunc_avg_irregular_hr + A_w_min_lr .* qfunc_avg_irregular_lr;
    end
    iEbN0 = iEbN0 + 1;
end

figure();

%semilogy(EbN0_min:EbN0_step:EbN0_max, fer, '-o', 'LineWidth',1.3);
%hold on;
semilogy(EbN0_min:EbN0_step:EbN0_max, fer_sim_qfunc, '-^', 'LineWidth',1.3);
hold on;
semilogy(EbN0_min:EbN0_step:EbN0_max, fer_sim_qfunc_irregular, '-o', 'LineWidth',1.3);
grid on;