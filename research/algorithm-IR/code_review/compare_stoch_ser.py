import numpy as np
d0 = np.load('code_review/snapshot_v0_pre_fix.npz', allow_pickle=True)
d1 = np.load('code_review/snapshot_v1_post_fix.npz', allow_pickle=True)
x_true = d0['_inputs_x_true']
print(f"x_true shape: {x_true.shape}")
print()
diffs = ['multi_restart','sa','random_walk','mh','importance_sampling','particle_filter']
print(f"{'detector':25s} {'v0 SER':>10} {'v1 SER':>10} {'v0 mae':>10} {'v1 mae':>10}")
for d in diffs:
    q0 = d0[f'xquant__{d}']
    q1 = d1[f'xquant__{d}']
    se0 = float((np.abs(q0 - x_true) > 1e-6).mean())
    se1 = float((np.abs(q1 - x_true) > 1e-6).mean())
    ne0 = float(np.mean(np.abs(d0[f'xhat__{d}'] - x_true)))
    ne1 = float(np.mean(np.abs(d1[f'xhat__{d}'] - x_true)))
    print(f"{d:25s} {se0:>10.4f} {se1:>10.4f} {ne0:>10.4f} {ne1:>10.4f}")
print()
print("= reference (bit-exact) =")
for d in ['lmmse','zf','amp']:
    q0 = d0[f'xquant__{d}']
    q1 = d1[f'xquant__{d}']
    se0 = float((np.abs(q0 - x_true) > 1e-6).mean())
    se1 = float((np.abs(q1 - x_true) > 1e-6).mean())
    print(f"{d:25s} {se0:>10.4f} {se1:>10.4f}")
