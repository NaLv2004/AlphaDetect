"""Probe: does RL/policy loss reach the encoder via _train_step?"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np, torch, logging
logging.basicConfig(level=logging.INFO)

from torch_geometric.data import Data
from evolution.gnn_pattern_matcher import GNNPatternMatcher, _NODE_DIM, _VALUE_FEAT_DIM

m = GNNPatternMatcher(buffer_size=64, train_steps=1, train_interval=1,
                     lambda_rl=1.0, mask_invalid_loss_weight=0.2,
                     entropy_coef=0.1, device='cpu')

w0_proj = m.encoder.node_proj.weight.detach().clone()
w0_conv = m.encoder.conv1.lin.weight.detach().clone()

NODE_DIM = _NODE_DIM
FEAT_DIM = _VALUE_FEAT_DIM
host_algo, donor_algo = 'algoA', 'algoB'
g_h = Data(x=torch.randn(5, NODE_DIM),
           edge_index=torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]], dtype=torch.long))
g_d = Data(x=torch.randn(5, NODE_DIM),
           edge_index=torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]], dtype=torch.long))
m._graph_cache[host_algo] = g_h
m._graph_cache[donor_algo] = g_d
m._visible_op_idx_cache[host_algo] = {f'op_{i}': i for i in range(5)}
m._visible_op_idx_cache[donor_algo] = {f'op_{i}': i for i in range(5)}
with torch.no_grad():
    e_h = m.encoder(g_h); e_d = m.encoder(g_d)
m._emb_cache[host_algo] = e_h
m._emb_cache[donor_algo] = e_d

sample_feats = np.random.randn(3, FEAT_DIM).astype(np.float32)


def make_exp(pid):
    return dict(
        proposal_id=pid, host_algo=host_algo, donor_algo=donor_algo,
        predicted_graft_score=1.0, generation=0,
        host_context_emb=np.zeros(32, dtype=np.float32),
        donor_context_emb=np.zeros(32, dtype=np.float32),
        host_output_candidates=['v_0', 'v_1', 'v_2'],
        host_output_feats=sample_feats,
        host_output_def_op_ids=['op_0', 'op_1', 'op_2'],
        host_output_trace=[{'available': [0, 1, 2], 'kept_local': [0, 1, 2],
                            'stop_added': False, 'chose_global': 1}],
        host_selected_outputs=['v_1'], host_effective_outputs=['v_1'],
        host_cut_candidates=[], host_cut_feats=np.zeros((0, FEAT_DIM), dtype=np.float32),
        host_cut_def_op_ids=[], host_cut_trace=[],
        host_selected_cuts=[], host_effective_cuts=[],
        host_region_validity='ok',
        donor_output_candidates=['v_0', 'v_1', 'v_2'],
        donor_output_feats=np.random.randn(3, FEAT_DIM).astype(np.float32),
        donor_selected_outputs=['v_1'], donor_effective_outputs=['v_1'],
        donor_cut_candidates=[], donor_cut_feats=np.zeros((0, FEAT_DIM), dtype=np.float32),
        donor_selected_cuts=[], donor_effective_cuts=[],
        donor_region_validity='ok',
        host_temperature=1.0, donor_temperature=1.0,
        contract_signature={},
    )


for pid in ('p1', 'p2', 'p3'):
    m._experience.append(make_exp(pid))
    m._outcomes[pid] = dict(reward=0.5, graft_score=1.0, host_score=None,
                            is_valid=True, host_algo=host_algo,
                            terminal_reasonable=True, behavior_change_rate=0.4)
m._total_proposals = 3

m._train_step(n_steps=1)

w1_proj = m.encoder.node_proj.weight.detach()
w1_conv = m.encoder.conv1.lin.weight.detach()
delta_proj = (w1_proj - w0_proj).abs().max().item()
delta_conv = (w1_conv - w0_conv).abs().max().item()
print('NODE_PROJ_DELTA=%.6e' % delta_proj)
print('CONV1_DELTA=%.6e' % delta_conv)
print('STATS=', m._last_train_stats)
assert delta_proj > 0, 'node_proj weights did not change'
assert delta_conv > 0, 'conv1 weights did not change'
print('OK encoder receives gradient from policy losses')
