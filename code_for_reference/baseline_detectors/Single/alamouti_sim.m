% =========================================================================
% XSimple AI Assistant - Refined MIMO Simulation
% Scenario: 2x2 MIMO
% Part 1: Fair comparison at Fixed Rate (2 bits/s/Hz)
% Part 2: Asymptotic DMT Tradeoff verification (Semi-Analytic)
% =========================================================================

clear; clc; close all;

%% =========================================================================
%% PART 1: SER vs SNR (Fair Comparison: Same Spectral Efficiency)
%% =========================================================================
% Alamouti (QPSK, Rate 1, 2 bits/sym) vs SM (BPSK, Rate 2, 1 bit/sym)

fprintf('Running Part 1: Fair SER comparison (Rate = 2 bits/s/Hz)...\n');

SNR_dB_vec = 0:2:20;
SNR_lin_vec = 10.^(SNR_dB_vec./10);
num_trials = 2000000; 

% --- Constellations (Unit Energy) ---
qpsk_syms = [1+1i, 1-1i, -1+1i, -1-1i] / sqrt(2); 
bpsk_syms = [1, -1]; 

% Pre-allocate
ser_sm = zeros(size(SNR_dB_vec));
ser_ala = zeros(size(SNR_dB_vec));

% SM Codebook for ML (BPSK x BPSK = 4 combinations)
SM_Codebook = [1, 1, -1, -1; 
               1, -1, 1, -1]; 

for s_idx = 1:length(SNR_dB_vec)
    snr_lin = SNR_lin_vec(s_idx);
    
    % Power Scaling
    scale_sm = sqrt(snr_lin / 2);
    scale_ala = sqrt(snr_lin / 2);
    
    err_sm = 0;
    err_ala = 0;
    
    for t = 1:num_trials
        % Channel 2x2 CN(0,1)
        H = (randn(2,2) + 1i*randn(2,2)) / sqrt(2);
        % Noise 2x1
        n = (randn(2,1) + 1i*randn(2,1)) / sqrt(2);
        
        %% 1. SM (V-BLAST) with ML Detection (BPSK)
        tx_idx = randi(4);
        x_vec = SM_Codebook(:, tx_idx);
        y_sm = scale_sm * H * x_vec + n;
        
        % ML Detector
        min_d = inf; est_idx = -1;
        for c = 1:4
            d = norm(y_sm - scale_sm * H * SM_Codebook(:, c))^2;
            if d < min_d, min_d = d; est_idx = c; end
        end
        if est_idx ~= tx_idx, err_sm = err_sm + 1; end
        
        %% 2. Alamouti (QPSK)
        idx1 = randi(4); s1 = qpsk_syms(idx1);
        % Alamouti Effective Scalar Channel Gain: ||H||_F^2
        h_pow = norm(H, 'fro')^2;
        n_eff = (randn + 1i*randn)/sqrt(2);
        
        % Effective Received Signal
        y_ala = sqrt(scale_ala^2 * h_pow) * s1 + n_eff;
        
        % Detection
        [~, det_i] = min(abs(y_ala - sqrt(scale_ala^2 * h_pow) * qpsk_syms));
        if det_i ~= idx1, err_ala = err_ala + 1; end
    end
    
    ser_sm(s_idx) = err_sm / num_trials;
    ser_ala(s_idx) = err_ala / num_trials;
end

%% =========================================================================
%% PART 2: Diversity-Multiplexing Tradeoff (Semi-Analytic Simulation)
%% =========================================================================
fprintf('\nRunning Part 2: Asymptotic DMT Tradeoff (Semi-Analytic)...\n');

% Use high SNR to extract asymptotic slope
SNR_dB_DMT = [60, 70, 80]; 
SNR_lin_DMT = 10.^(SNR_dB_DMT./10);

r_vals = 0:0.1:1; 
d_simulated = zeros(size(r_vals));

for r_idx = 1:length(r_vals)
    r = r_vals(r_idx);
    
    % 【关键修改】：为了计算 r=0 处的极限值，使用一个极小正数代替 0
    % 否则 target_rate = 0，中断概率恒为0，导致斜率无法计算
    r_calc = max(r, 1e-5); 
    
    p_out = zeros(size(SNR_dB_DMT));
    
    for s_i = 1:length(SNR_dB_DMT)
        snr = SNR_lin_DMT(s_i);
        
        % Threshold derivation:
        % log2(1 + (SNR/2)*X) < r * log2(SNR)
        % 1 + (SNR/2)*X < SNR^r
        % X < 2 * (SNR^r - 1) / SNR
        
        % 使用 r_calc 代替 r 进行计算
        threshold_X = 2 * (snr^r_calc - 1) / snr;
        
        % Gamma CDF: 2x2 MIMO ||H||_F^2 ~ Gamma(4, 1)
        p_out(s_i) = gamcdf(threshold_X, 4, 1); 
        
        % Avoid log(0)
        if p_out(s_i) == 0, p_out(s_i) = 1e-100; end
    end
    
    % Extract Diversity Gain d
    coeffs = polyfit(log10(SNR_lin_DMT), log10(p_out), 1);
    d_simulated(r_idx) = -coeffs(1);
    
    fprintf('r = %.1f, Simulated d = %.3f\n', r, d_simulated(r_idx));
end

d_theory = 4 * (1 - r_vals);

%% PLOTTING (Part 2 Only for clarity)
% figure;


%% =========================================================================
%% PLOTTING
%% =========================================================================

figure('Position', [100, 100, 1000, 500]);

% Plot 1: SER
subplot(1, 2, 1);
semilogy(SNR_dB_vec, ser_sm, 'b-o', 'LineWidth', 2); hold on;
semilogy(SNR_dB_vec, ser_ala, 'r-s', 'LineWidth', 2);
grid on;
xlabel('SNR (dB)');
ylabel('Symbol Error Rate (SER)');
title({'Fair Comparison (Rate = 2 bits/s/Hz)', 'Alamouti(QPSK) vs SM(BPSK)'});
legend('2x2 SM (BPSK)', '2x2 Alamouti (QPSK)', 'Location', 'SouthWest');

% Plot 2: DMT
subplot(1, 2, 2);
plot(r_vals, d_theory, 'k--', 'LineWidth', 2); hold on;
plot(r_vals, d_simulated, 'bo-', 'LineWidth', 2, 'MarkerFaceColor', 'b');
grid on;
xlabel('Multiplexing Gain r');
ylabel('Diversity Gain d');
title({'Diversity-Multiplexing Tradeoff', 'Alamouti Scheme (2x2) - Corrected'});
legend('Theoretical d(r) = 4(1-r)', 'Simulated (Semi-Analytic)', 'Location', 'NorthEast');
axis([0 1 0 4.5]);