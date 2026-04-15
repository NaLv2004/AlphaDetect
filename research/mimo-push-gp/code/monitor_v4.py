"""Monitor v4 experiments for BP emergence.
Run with: python -B monitor_v4.py
"""
import os
import time
import re
import glob

LOG_DIR = r'd:\ChannelCoding\RCOM\AlphaDetect\research\mimo-push-gp\logs'
PATTERNS = ['bp_evolution_v4_*.log']

def get_gen_lines(log_file):
    """Extract generation lines from log."""
    lines = []
    try:
        with open(log_file, 'r') as f:
            for line in f:
                if re.search(r'Gen \d+.*Fit\(', line):
                    lines.append(line.strip())
    except:
        pass
    return lines

def parse_bp(line):
    """Extract bp= value from gen line."""
    m = re.search(r'bp=([\d.]+)', line)
    return float(m.group(1)) if m else 0.0

def parse_ber(line):
    """Extract BER from line."""
    m = re.search(r'BER=([\d.]+)', line)
    return float(m.group(1)) if m else 1.0

def parse_ratio(line):
    """Extract ratio from line."""
    m = re.search(r'ratio=([\d.]+)', line)
    return float(m.group(1)) if m else 1.0

def main():
    print("Monitoring v4 experiments for BP emergence...")
    print("Looking for: bp > 0, ratio < 0.5, bidirectional multi-layer patterns")
    print()
    
    log_files = []
    for pat in PATTERNS:
        log_files.extend(glob.glob(os.path.join(LOG_DIR, pat)))
    
    if not log_files:
        print("No log files found yet.")
        return
    
    for lf in sorted(log_files):
        name = os.path.basename(lf)
        gen_lines = get_gen_lines(lf)
        if not gen_lines:
            print(f"{name}: (no generations yet)")
            continue
        
        print(f"\n{'='*60}")
        print(f"Experiment: {name}")
        print(f"{'='*60}")
        
        best_bp = 0.0
        best_ber = 1.0
        best_ratio = 1.0
        bp_emerged_gen = None
        best_gen_line = None
        
        for i, line in enumerate(gen_lines):
            bp = parse_bp(line)
            ber = parse_ber(line)
            ratio = parse_ratio(line)
            
            if bp > 0 and bp_emerged_gen is None:
                bp_emerged_gen = i
                
            if ber < best_ber:
                best_ber = ber
                best_gen_line = line
                
            if bp > best_bp:
                best_bp = bp
        
        print(f"Generations: {len(gen_lines)}")
        print(f"Best BER: {best_ber:.5f}")
        print(f"BP first emerged: gen {bp_emerged_gen if bp_emerged_gen else 'NOT YET'}")
        print(f"Best bp_updates: {best_bp:.1f}")
        
        if best_gen_line:
            print(f"\nBest gen: {best_gen_line[:120]}")
            
        if bp_emerged_gen is not None:
            print(f"\n{'!'*40}")
            print(f"BP ACTIVE! Getting bp_updates > 0 from gen {bp_emerged_gen}")
            print(f"{'!'*40}")
        
        # Show last 5 gen lines
        print(f"\nLast {min(5, len(gen_lines))} generations:")
        for line in gen_lines[-5:]:
            bp = parse_bp(line)
            marker = " *** BP!" if bp > 0 else ""
            print(f"  {line[:100]}{marker}")
        
        # Show best BP gen if exists  
        best_bp_gens = [(i, line) for i, line in enumerate(gen_lines) 
                        if parse_bp(line) > 10]
        if best_bp_gens:
            print(f"\nGenerations with significant BP (>10 updates):")
            for i, line in best_bp_gens[-3:]:
                print(f"  Gen {i}: {line[:120]}")

if __name__ == '__main__':
    main()
