clc; 
clear;

Nt = 32; % n of transmitting antennas
Nr = 64; % n of receiving antennas
EbN0_min = 5;
EbN0_max = 10;
EbN0_step = 1;
ModType = 1;
CodeRate = 676 ./ 1024;
EbN0_len = round((EbN0_max - EbN0_min) /EbN0_step) + 1;
fer = zeros(EbN0_len ,1);
fer_sim_qfunc = zeros(EbN0_len, 1);
n_sim_channels = 100;
n_sim_qfunc = 1000000;
w_min = 16;    % minimum weight
A_w_min = 1240 .* 1240;    % n of minimum weight codewords
atten = 2;
S_square_stored = zeros(n_sim_channels *Nt* 2, 1);

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
    qfunc_avg = 1000.0;
    qfunc_sum = 0.0;
    singular_values_sum = 0.0;
    for i_sim_channel = 1:n_sim_channels
        H = randn(Nr, Nt);
        [U,S,V] = svd(H);
        S = svd(H);
        S_square = S .^ 2;
        S_square_stored((i_sim_channel-1)*Nt+1 : i_sim_channel*Nt) = S_square;
        singular_values_sum = singular_values_sum + sum(S_square, 'all');
        % qfunc_tmp = qfunc(sqrt(w_min .* tmp .* CodeRate .* 2));
        % qfunc(sqrt( w_min .* h2_avg ./ (sigma_sq.^2)));
        % qfunc_sum = qfunc_sum + qfunc_tmp;
    end
    for i_sim_qfunc = 1 : n_sim_qfunc
        w_min_half = round(w_min ./ 2);
        selected_pos = randi([1,n_sim_channels*Nt],w_min_half,1);
        selected_singular_square = S_square_stored(selected_pos);
        sum_tmp = sum(selected_singular_square).*2;
        qfunc_tmp = qfunc(sqrt(sum_tmp./ (sigma_sq.^2)));
        % qfunc_tmp
        qfunc_sum = qfunc_sum + qfunc_tmp;
        if (qfunc_tmp < qfunc_avg) 
            qfunc_avg = qfunc_tmp;
        end
    end
    singular_values_avg = singular_values_sum ./ (Nt .* n_sim_channels);
    h2_avg = singular_values_avg;
    % qfunc_avg = qfunc_sum ./ n_sim_qfunc;
    if (atten == 1)
        % sqrt(2 .* w_min .* 1./ sigma_sq)
        % qfunc(sqrt(w_min .* 1./ sigma_sq))
        % fer (iEbN0, 1) = A_w_min .* qfunc(sqrt(w_min .* 1./ sigma_sq));
        fer (iEbN0, 1) = A_w_min .* qfunc(sqrt(w_min .* tmp .* CodeRate .* 2));
    else
        fer (iEbN0, 1) = A_w_min .* qfunc(sqrt( w_min .* h2_avg ./ (sigma_sq.^2)));
        fer_sim_qfunc (iEbN0, 1) = A_w_min .* qfunc_avg;
    end
    iEbN0 = iEbN0 + 1;
end

figure();

% semilogy(EbN0_min:EbN0_step:EbN0_max, fer, '-o', 'LineWidth',1.3);
% hold on;
semilogy(EbN0_min:EbN0_step:EbN0_max, fer_sim_qfunc, '-^', 'LineWidth',1.3);
grid on;