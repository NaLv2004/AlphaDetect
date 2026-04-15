REM cpp_test5: Warm-start from cpp_test4 gen-4 best genome.
REM Key changes from test4:
REM   - Seed from gen-4 best genome (hall-of-fame warm start)
REM   - 60 generations (more exploration budget)
REM   - max_nodes increased to 1200 (forces larger-tree optimization)
REM   - Same training params otherwise
REM   - Hard restart at stagnation >= 12 gens (already in code)
REM   - flops_max increased proportionally (1,200,000 for 1200 nodes)
python -u -B bp_main_v2.py --generations 60 --population 300 --train-samples 80 --train-max-nodes 1200 --train-flops-max 1200000 --step-max 1000 --train-snrs "10,12,14" --eval-trials 200 --eval-max-nodes 2000 --eval-flops-max 5000000 --eval-step-max 5000 --train-nt 8 --train-nr 16 --log-suffix cpp_test5 --use-cpp --seed-genome-json seed_genomes\cpp_test4_gen4_best.json
