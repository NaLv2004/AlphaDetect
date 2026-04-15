Set-Location d:\ChannelCoding\RCOM\AlphaDetect\research\mimo-push-gp\code
conda run -n AutoGenOld python -u -B main.py --population 64 --generations 12 --train-samples 10 --train-nt 16 --train-nr 32 --eval-nt 16 --eval-nr 32 --mod-order 16 --train-max-nodes 320 --train-flops-max 120000 --eval-trials 24 --eval-max-nodes 2500 --eval-flops-max 300000 --train-snrs 12,16,20 --snrs 12,16,20 *>&1
