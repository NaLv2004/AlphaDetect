function [symLLR, alpha] = Bsp_dm_df(TxAntNum, RxAntNum, slen, RxSymbol, H, Nv, sym, delta, iterNum, dm, df)
% Bsp_dm_df: Belief Selective Propagation MIMO Detector (Fixed & Optimized)
% 性能已修复：增加了严格的 Beta 归一化和数值稳定处理，匹配 Bsp_b11_fast 的性能。

    % --- 1. Initialization ---
    [Nr_real, Nt_real] = size(H); 
    
    sym = sym(:).';       % [1 x M]
    RxSymbol = RxSymbol(:); % [Nr x 1]
    
    % LMMSE Initialization
    % W = (H'H + Nv*I)^-1 H'
    W = (H' * H + Nv * eye(Nt_real)) \ H';
    s_mmse = W * RxSymbol;
    
    % Initial Alpha (VN -> CN)
    % 使用欧氏距离平方 (Standard Gaussian Likelihood)
    dist_sq = abs(repmat(s_mmse, 1, slen) - repmat(sym, Nt_real, 1)).^2;
    alpha_init_val = -dist_sq / (2 * Nv); 
    
    % [Crucial] Normalize Initialization to prevent "locking" too early
    alpha_init_val = alpha_init_val - max(alpha_init_val, [], 2);
    
    % Broadcast to edges: [Nt, Nr, M]
    alpha = repmat(reshape(alpha_init_val, [Nt_real, 1, slen]), 1, Nr_real, 1);
    
    % Initialize Beta (CN -> VN)
    beta = zeros(Nt_real, Nr_real, slen);
    
    % Pre-sort H for Selective Strategy
    [~, H_sort_idx] = sort(abs(H), 2, 'descend'); 
    
    % Pre-compute combinations indices if df > 1
    num_chosen_edges = max(0, df - 1);
    if num_chosen_edges > 0
        comb_indices = combinator_indices(dm, num_chosen_edges);
    else
        comb_indices = [];
    end
    
    % --- 2. Iterative Loop ---
    for iter = 1:iterNum
        
        % Normalize Alpha (Input for this iteration)
        alpha = alpha - max(alpha, [], 3);
        
        % --- Step A: Symbol Sorting & Truncation ---
        % Find top-dm candidates for each user
        [alpha_sorted, alpha_sort_indices] = sort(alpha, 3, 'descend');
        
        % Indices: [Nt, Nr, dm]
        top_dm_indices = alpha_sort_indices(:, :, 1:min(dm, slen));
        top_dm_alphas  = alpha_sorted(:, :, 1:min(dm, slen));
        
        % Hard decision (Top-1) for simplified interference cancellation
        hard_sym_idx = top_dm_indices(:, :, 1);
        hard_sym_val = sym(hard_sym_idx); % [Nt, Nr]
        
        % --- Step B: Update Beta (CN i -> VN j) ---
        for i = 1:Nr_real
            neighbors_sorted = H_sort_idx(i, :);
            
            for j = 1:Nt_real
                % Identify neighbors
                neighbors_ex_j = neighbors_sorted(neighbors_sorted ~= j);
                
                % Split into Chosen (Combinatorial) and Simplified (Hard Cancel)
                if df > 1
                    len_ne = length(neighbors_ex_j);
                    cut = min(len_ne, df-1);
                    idx_chosen = neighbors_ex_j(1 : cut);
                    idx_simp   = neighbors_ex_j(cut+1 : end);
                else
                    idx_chosen = [];
                    idx_simp   = neighbors_ex_j;
                end
                
                % 1. Cancel Simplified Interference (using Hard Decisions)
                if ~isempty(idx_simp)
                    % Note: hard_sym_val(k, i) is the best guess of k sent to i
                    simp_vals = hard_sym_val(idx_simp, i); 
                    I_simp = sum(H(i, idx_simp).' .* simp_vals);
                else
                    I_simp = 0;
                end
                
                y_eff = RxSymbol(i) - I_simp;
                
                % 2. Combinatorial Search over Chosen Edges
                if isempty(idx_chosen)
                    % --- Case df=1 (Standard BP / Interference Cancellation) ---
                    % Residual = y_eff - H(i,j)*s_j
                    res = y_eff - H(i, j) * sym; 
                    metric = -abs(res).^2 / (2 * Nv);
                    
                else
                    % --- Case df > 1 (Belief Selective) ---
                    num_combs = size(comb_indices, 1);
                    
                    I_chosen_combs = zeros(1, num_combs);
                    Alpha_sum_combs = zeros(1, num_combs);
                    
                    for k_idx = 1:length(idx_chosen)
                        k = idx_chosen(k_idx);
                        
                        % Extract candidates for neighbor k
                        % Use squeeze to ensure [dm x 1] vector
                        curr_idxs = squeeze(top_dm_indices(k, i, :)); 
                        c_syms    = sym(curr_idxs).';       % [dm x 1]
                        c_alphas  = squeeze(top_dm_alphas(k, i, :)); % [dm x 1]
                        
                        % Map based on combinations
                        col_idx = comb_indices(:, k_idx);
                        vals = c_syms(col_idx); 
                        als  = c_alphas(col_idx);
                        
                        I_chosen_combs  = I_chosen_combs + H(i, k) * vals.';
                        Alpha_sum_combs = Alpha_sum_combs + als.';
                    end
                    
                    % Calculate Metric for all hypotheses of symbol s_j
                    % Res: [M x num_combs]
                    % y_eff - I_chosen - H(i,j)*s_j
                    Res = (y_eff - I_chosen_combs) - (H(i, j) * sym(:));
                    
                    Euc = -abs(Res).^2 / (2 * Nv);
                    
                    % Add priors from chosen neighbors
                    TotalMetric = Euc + repmat(Alpha_sum_combs, slen, 1);
                    
                    % Max-Log approximation (marginalize over combinations)
                    metric = max(TotalMetric, [], 2).'; % [1 x M]
                end
                
                % [CRITICAL FIX] Normalize Beta immediately!
                % Subtract max to keep values in a reasonable numerical range (e.g., [-20, 0])
                % This matches the stability of "beta - beta(:,:,1)" in the fast version.
                beta(j, i, :) = metric - max(metric);
                
            end 
        end 
        
        % --- Step C: Update Alpha (VN j -> CN i) ---
        % Sum incoming betas (Log-domain product)
        beta_sum = sum(beta, 2); % [Nt, 1, M]
        
        % Extrinsic Information
        alpha_new = repmat(beta_sum, 1, Nr_real, 1) - beta;
        
        % [CRITICAL FIX] Normalize Alpha_new before damping
        % If we don't normalize here, alpha_new might have a different offset than alpha,
        % making the damping (weighted average) meaningless.
        alpha_new = alpha_new - max(alpha_new, [], 3);
        
        % Damping Update
        alpha = (1 - delta) * alpha_new + delta * alpha;
        
    end 
    
    % Final Output
    symLLR = squeeze(sum(beta, 2));
    % Final Normalization for output
    symLLR = symLLR - max(symLLR, [], 2);

end

function indices = combinator_indices(dm, num_vars)
    if num_vars == 0
        indices = [];
        return;
    end
    args = repmat({1:dm}, 1, num_vars);
    [G{1:num_vars}] = ndgrid(args{:});
    indices = zeros(dm^num_vars, num_vars);
    for i = 1:num_vars
        indices(:, i) = G{i}(:);
    end
end