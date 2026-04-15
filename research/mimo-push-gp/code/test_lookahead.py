"""
Test the theoretically optimal one-step look-ahead correction h_{k-1}.

h_{k-1} = min_{s in Omega} |y'_{k-1} - R_{k-1,k-1} * s|^2

where y'_{k-1} = y_tilde_{k-1} - sum_{j=k}^{Nt-1} R_{k-1,j} * x_j

We compare:
1. No correction (pure distance ordering)  
2. Discovered correction (local_dist + Re(symbol))
3. Optimal one-step look-ahead
"""
import numpy as np
import time
from vm import MIMOPushVM, Instruction
from stack_decoder import (StackDecoder, lmmse_detect, kbest_detect,
                           qam16_constellation)
from stacks import SearchTreeGraph, TreeNode

Nt, Nr = 8, 16
constellation = qam16_constellation()


def generate_sample(snr_db, rng):
    H = (rng.randn(Nr, Nt) + 1j * rng.randn(Nr, Nt)) / np.sqrt(2)
    x_idx = rng.randint(0, len(constellation), Nt)
    x = constellation[x_idx]
    sig_power = float(np.mean(np.abs(H @ x) ** 2))
    noise_var = sig_power / (10 ** (snr_db / 10.0))
    noise = np.sqrt(noise_var / 2) * (rng.randn(Nr) + 1j * rng.randn(Nr))
    y = H @ x + noise
    return H, x, y, noise_var


def ber_calc(x_true, x_hat):
    return float(np.mean(x_true != x_hat))


# --- Optimal look-ahead scoring function ---
class LookAheadDecoder:
    """Stack decoder with optimal one-step look-ahead h_{k-1}."""
    
    def __init__(self, Nt, Nr, constellation, max_nodes):
        self.Nt = Nt
        self.Nr = Nr
        self.constellation = constellation
        self.M = len(constellation)
        self.max_nodes = max_nodes
    
    def detect(self, H, y, mode='lookahead'):
        """
        mode: 'distance' = no correction
              'discovered' = local_dist + Re(symbol)  
              'lookahead' = optimal h_{k-1}
        """
        Q, R = np.linalg.qr(H, mode='reduced')
        y_tilde = Q.conj().T @ y  # complex!
        
        import heapq
        
        counter = 0
        pq = []
        
        # Root: layer Nt-1
        root_layer = self.Nt - 1
        for sym in self.constellation:
            residual = y_tilde[root_layer] - R[root_layer, root_layer] * sym
            ld = float(np.abs(residual) ** 2)
            cd = ld
            partial = [sym]  # store complex symbols
            
            # Compute correction
            if mode == 'distance':
                correction = 0.0
            elif mode == 'discovered':
                correction = ld + float(np.real(sym))
            elif mode == 'lookahead':
                correction = self._lookahead(root_layer, partial, R, y_tilde)
            
            score = cd + correction
            counter += 1
            heapq.heappush(pq, (score, counter, root_layer - 1, cd, ld, partial, sym))
        
        nodes_used = len(pq)
        
        while pq and nodes_used < self.max_nodes:
            score, _, layer, cd, ld, partial, sym = heapq.heappop(pq)
            
            if layer < 0:
                # Complete path found
                x_hat = np.array(partial[::-1])
                return x_hat, nodes_used
            
            # Expand: add children at current layer
            for sym_new in self.constellation:
                # Compute interference from all decided symbols
                interference = 0.0 + 0.0j
                for j in range(len(partial)):
                    col = self.Nt - 1 - j
                    interference += R[layer, col] * partial[j]
                residual = y_tilde[layer] - R[layer, layer] * sym_new - interference
                ld_new = float(np.abs(residual) ** 2)
                cd_new = cd + ld_new
                partial_new = partial + [sym_new]
                
                if mode == 'distance':
                    correction = 0.0
                elif mode == 'discovered':
                    correction = ld_new + float(np.real(sym_new))
                elif mode == 'lookahead':
                    correction = self._lookahead(layer, partial_new, R, y_tilde)
                
                score_new = cd_new + correction
                counter += 1
                nodes_used += 1
                heapq.heappush(pq, (score_new, counter, layer - 1, cd_new, ld_new,
                                    partial_new, sym_new))
                
                if nodes_used >= self.max_nodes:
                    break
        
        # Budget exhausted — find best partial/complete path
        if pq:
            best = heapq.heappop(pq)
            partial = best[5]
            # Greedily complete the path with minimum-distance decisions
            x_hat = self._greedy_complete(partial, R, y_tilde)
            return x_hat, nodes_used
        
        return np.zeros(self.Nt, dtype=complex), nodes_used
    
    def _greedy_complete(self, partial, R, y_tilde):
        """Greedily extend partial path to full path with min local_dist at each layer."""
        symbols = list(partial)  # decided symbols so far
        n_decided = len(symbols)
        next_layer = self.Nt - 1 - n_decided
        
        for layer in range(next_layer, -1, -1):
            best_sym = None
            best_dist = float('inf')
            for sym in self.constellation:
                interference = 0.0 + 0.0j
                for j in range(len(symbols)):
                    col = self.Nt - 1 - j
                    interference += R[layer, col] * symbols[j]
                residual = y_tilde[layer] - R[layer, layer] * sym - interference
                d = float(np.abs(residual) ** 2)
                if d < best_dist:
                    best_dist = d
                    best_sym = sym
            symbols.append(best_sym)
        
        return np.array(symbols[::-1])
    
    def _lookahead(self, layer, partial_symbols, R, y_tilde):
        """
        Compute h_{k-1}: the minimum achievable distance at the NEXT undecided layer.
        
        h_{k-1} = min_{s in Omega} |y'_{k-1} - R_{k-1,k-1} * s|^2
        where y'_{k-1} = y_tilde_{k-1} - sum_{j=k}^{Nt-1} R_{k-1,j} * x_j
        """
        k = layer  # current layer
        next_layer = k - 1
        
        if next_layer < 0:
            return 0.0  # no more layers
        
        # Compute interference-cancelled observation (complex!)
        y_prime = y_tilde[next_layer]
        for j in range(len(partial_symbols)):
            actual_layer = self.Nt - 1 - j
            y_prime = y_prime - R[next_layer, actual_layer] * partial_symbols[j]
        
        # Find minimum distance over constellation  
        diag = R[next_layer, next_layer]
        min_dist = float('inf')
        for sym in self.constellation:
            d = float(np.abs(y_prime - diag * sym) ** 2)
            if d < min_dist:
                min_dist = d
        
        return min_dist


# --- Run comparison ---
def run_comparison(n_trials=200, snr_dbs=None, max_nodes=60):
    if snr_dbs is None:
        snr_dbs = [8, 10, 12, 14, 16, 18, 20]
    
    rng = np.random.RandomState(2026)
    decoder = LookAheadDecoder(Nt, Nr, constellation, max_nodes)
    
    print(f"{'SNR':>5}  {'Distance':>12}  {'Discovered':>12}  "
          f"{'LookAhead':>12}  {'LMMSE':>12}  {'KB16':>12}  {'KB32':>12}")
    print("-" * 90)
    
    for snr in snr_dbs:
        bers = {'distance': [], 'discovered': [], 'lookahead': [],
                'lmmse': [], 'kb16': [], 'kb32': []}
        
        for _ in range(n_trials):
            H, x, y, nv = generate_sample(snr, rng)
            
            for mode in ['distance', 'discovered', 'lookahead']:
                xh, _ = decoder.detect(H, y, mode=mode)
                bers[mode].append(ber_calc(x, xh))
            
            xl, _ = lmmse_detect(H, y, nv, constellation)
            bers['lmmse'].append(ber_calc(x, xl))
            
            xk16, _ = kbest_detect(H, y, constellation, K=16)
            bers['kb16'].append(ber_calc(x, xk16))
            
            xk32, _ = kbest_detect(H, y, constellation, K=32)
            bers['kb32'].append(ber_calc(x, xk32))
        
        means = {k: np.mean(v) for k, v in bers.items()}
        print(f"{snr:5.0f}  {means['distance']:12.5f}  {means['discovered']:12.5f}  "
              f"{means['lookahead']:12.5f}  {means['lmmse']:12.5f}  "
              f"{means['kb16']:12.5f}  {means['kb32']:12.5f}")


if __name__ == '__main__':
    print("=== max_nodes=60 (training budget) ===")
    t0 = time.time()
    run_comparison(n_trials=100, max_nodes=60)
    print(f"\n  Time: {time.time()-t0:.1f}s")
    
    print("\n=== max_nodes=1500 (eval budget) ===")
    t0 = time.time()
    run_comparison(n_trials=100, max_nodes=1500)
    print(f"\n  Time: {time.time()-t0:.1f}s")
