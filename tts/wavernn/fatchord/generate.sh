#gpui python gen_wavernn.py --voc_weights latest_weights.pyt --samples 50
gpui python gen_wavernn.py --voc_weights latest_weights.pyt --samples 50 --unbatched

#srun -p train --cpus-per-task=12 -t 128:00:00 --gpus-per-node=1 \
#    python gen_wavernn.py --voc_weights latest_weights.pyt --samples 50 --unbatched
