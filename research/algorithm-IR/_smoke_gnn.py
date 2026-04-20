"""Smoke test for new GNN components."""
import sys
sys.path.insert(0, r'd:\ChannelCoding\RCOM\AlphaDetect\research\algorithm-IR')
from evolution.gnn_pattern_matcher import DonorRegionSelectorGNN, build_trimmed_donor_ir, GNNPatternMatcher
import torch
import torch.nn.functional as F

# Test DonorRegionSelectorGNN
gnn = DonorRegionSelectorGNN()
h = torch.randn(6, 27)
d = torch.randn(15, 27)
logits = gnn(h, d)
print(f'DonorRegionSelectorGNN: h={h.shape} d={d.shape} -> logits={logits.shape}')
probs = F.softmax(logits, dim=0)
idx = torch.multinomial(probs, 1).item()
print(f'Sampled start_idx={idx}/{d.shape[0]}')

# Test edge case: single op donor
d_small = torch.randn(1, 27)
logits2 = gnn(h, d_small)
print(f'Single-op donor: logits={logits2.shape}')

# Test empty host features
h_empty = torch.zeros(0, 27)
logits3 = gnn(h_empty, d)
print(f'Empty host: logits={logits3.shape}')

# Test GNNPatternMatcher construction
matcher = GNNPatternMatcher()
print(f'GNNPatternMatcher created, device={matcher.device}')
print(f'donor_region_selector params: {sum(p.numel() for p in matcher.donor_region_selector.parameters())}')
print(f'total params: {sum(p.numel() for name, p in zip(range(100), matcher._all_params))}')

# Test reward baseline initialization
print(f'reward_baseline={matcher._reward_baseline}')
print(f'baseline_alpha={matcher._baseline_alpha}')

print('ALL SMOKE TESTS PASSED')
