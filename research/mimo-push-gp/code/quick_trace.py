"""Quick Part 2: Step-by-step trace of Gen72 with fixed VM"""
import sys, os, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from analyze_truebp1 import explain_program
explain_program(snr_db=12, seed=999)
