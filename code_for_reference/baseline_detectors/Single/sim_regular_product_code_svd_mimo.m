clc; 
clear;

Nt = 32; % n of transmitting antennas
Nr = 64; % n of receiving antennas
EbN0_min = 2;
EbN0_max = 10;
EbN0_step = 1;
ModType = 4; % prev: 2
scaling_qam = sqrt(6./(2.^ModType-1)./2);
if (ModType==8)
    scaling_qam = 1;
end
CodeRate = 676 ./ 1024;
EbN0_len = round((EbN0_max - EbN0_min) /EbN0_step) + 1;

fer = zeros(EbN0_len ,1);
fer_sim_qfunc = zeros(EbN0_len, 1);
fer_sim_qfunc_min = zeros(EbN0_len, 1);
fer_sim_average_power = zeros(EbN0_len, 1);

n_sim_channels = 100;
n_sim_qfunc = 1000000;
w_min = 16;    % minimum weight
A_w_min = 1240 .* 1240;    % n of minimum weight codewords
N_min_channel = w_min;
atten = 2;
S_square_stored = zeros(n_sim_channels *Nt* 2, 1);
S_square_stored_min = zeros(N_min_channel*Nt*2, 1);
average_power = 0;

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
    qfunc_sum_min = 0.0;
    singular_values_sum = 0.0;
    for i_sim_channel = 1:n_sim_channels
        H = randn(Nr, Nt).*sqrt(0.5)+1i*randn(Nr, Nt).*sqrt(0.5);
        [U,S,V] = svd(H);
        S = svd(H);
        S_square = S .^ 2;
        % for test
        %sigma_sq = 2.73;
        prod = 1.0;
        for i = 1:Nt
            prod = prod.*(1+S_square(i) ./ sigma_sq);
        end
        prod_root = nthroot(prod, Nt);
        p_avg = (prod_root-1).*sigma_sq;
        gain_avg = sqrt(p_avg);
        S_square_stored((i_sim_channel-1)*Nt+1 : i_sim_channel*Nt) = S_square;
        S_square_stored_min((i_sim_channel-1)*N_min_channel+1: i_sim_channel* N_min_channel) = S_square(end-N_min_channel+1:end);
        singular_values_sum = singular_values_sum + sum(S_square, 'all');

        one_over_S_squared = 1./ S_square;
        average_power = Nt./ sum(one_over_S_squared);
        % qfunc_tmp = qfunc(sqrt(w_min .* tmp .* CodeRate .* 2));
        % qfunc(sqrt( w_min .* h2_avg ./ (sigma_sq.^2)));
        % qfunc_sum = qfunc_sum + qfunc_tmp;
    end
    for i_sim_qfunc = 1 : n_sim_qfunc
        w_min_half = round(w_min ./ 2);
        selected_pos = randi([1,n_sim_channels*Nt],w_min,1);
        selected_pos_min = randi([1,N_min_channel*Nt],w_min,1);
        selected_singular_square = S_square_stored(selected_pos);
        selected_sv_min = S_square_stored_min(selected_pos_min);
        sum_tmp = sum(selected_singular_square);
        qfunc_tmp = qfunc(sqrt(sum_tmp./ (sigma_sq))*scaling_qam);
        sum_tmp_min = sum(selected_sv_min);
        qfunc_tmp_min = qfunc(sqrt(sum_tmp_min./ (sigma_sq))*scaling_qam);
        qfunc_sum_min = qfunc_sum_min + qfunc_tmp_min;
        % qfunc_tmp
        qfunc_sum = qfunc_sum + qfunc_tmp;
    end
    singular_values_avg = singular_values_sum ./ (Nt .* n_sim_channels);
    h2_avg = singular_values_avg;
    qfunc_avg = qfunc_sum ./ n_sim_qfunc;
    qfunc_avg_min = qfunc_sum_min ./ n_sim_qfunc;
    if (atten == 1)
        % sqrt(2 .* w_min .* 1./ sigma_sq)
        % qfunc(sqrt(w_min .* 1./ sigma_sq))
        % fer (iEbN0, 1) = A_w_min .* qfunc(sqrt(w_min .* 1./ sigma_sq));
        fer (iEbN0, 1) = A_w_min .* qfunc(sqrt(w_min .* tmp .* CodeRate .* 2));
    else
        fer (iEbN0, 1) = A_w_min .* qfunc(sqrt( w_min .* h2_avg ./ (sigma_sq)));
        fer_sim_qfunc (iEbN0, 1) = A_w_min .* qfunc_avg;
        fer_sim_average_power (iEbN0, 1) = A_w_min .* qfunc(sqrt( w_min .* average_power ./ (sigma_sq))* scaling_qam);
        fer_sim_qfunc_min(iEbN0, 1) = A_w_min.*qfunc_avg_min;
    end
    iEbN0 = iEbN0 + 1;
end

figure();

%semilogy(EbN0_min:EbN0_step:EbN0_max, fer, '-o', 'LineWidth',1.3);
%hold on;
semilogy(EbN0_min:EbN0_step:EbN0_max, fer_sim_qfunc, '-^', 'LineWidth',1.3);
hold on;
semilogy(EbN0_min:EbN0_step:EbN0_max, fer_sim_average_power, '-o', 'LineWidth',1.3);
semilogy(EbN0_min:EbN0_step:EbN0_max, fer_sim_qfunc_min, '-s', 'LineWidth',1.3);
grid on;