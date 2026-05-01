clc; 
clear;

Nt = 32; % n of transmitting antennas
Nr = 64; % n of receiving antennas
EbN0_min = 1;
EbN0_max = 10;
EbN0_step = 1;
ModType = 2;
CodeRate = 676 ./ 1024;
EbN0_len = round((EbN0_max - EbN0_min) /EbN0_step) + 1;
fer = zeros(EbN0_len ,1);
n_sim_channels = 10;
w_min = 16;    % minimum weight
A_w_min = 1240 .* 1240;    % n of minimum weight codewords
atten = 1;

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
    singular_values_sum = 0.0;
    for i_sim_channel = 1:n_sim_channels
        H = randn(Nr, Nt).*sqrt(0.5)+1i*randn(Nr, Nt).*sqrt(0.5);
        [U,S,V] = svd(H);
        S_square = S .^ 2;
        singular_values_sum = singular_values_sum + sum(S_square, 'all');
        % qfunc_tmp = qfunc(sqrt(w_min .* tmp .* CodeRate .* 2));
        % qfunc(sqrt( w_min .* h2_avg ./ (sigma_sq.^2)));
        % qfunc_sum = qfunc_sum + qfunc_tmp;
    end
    singular_values_avg = singular_values_sum ./ (Nt .* n_sim_channels);
    h2_avg = singular_values_avg;
    qfunc_avg = qfunc_sum ./ n_sim_channels;
    if (atten == 1)
        % sqrt(2 .* w_min .* 1./ sigma_sq)
        % qfunc(sqrt(w_min .* 1./ sigma_sq))
        % fer (iEbN0, 1) = A_w_min .* qfunc(sqrt(w_min .* 1./ sigma_sq));
        w_min .* tmp .* CodeRate .* 2
        fer (iEbN0, 1) = A_w_min .* qfunc(sqrt(w_min .* tmp .* CodeRate .* 2));
    else
        fer (iEbN0, 1) = A_w_min .* qfunc(sqrt( w_min .* h2_avg ./ (sigma_sq.^2)));
        fer (iEbN0, 1) = A_w_min .* qfunc_avg;
    end
    iEbN0 = iEbN0 + 1;
end

figure();

semilogy(EbN0_min:EbN0_step:EbN0_max, fer, '-o', 'LineWidth',1.3);

grid on;