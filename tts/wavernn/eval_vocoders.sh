CUDA_VISIBLE_DEVICES=1 python eval_wavernn_nvidia.py --checkpoint ../../tts-subjective/models/parallel_wavernn_nvidia_ckpt_bs32.pt

#CUDA_VISIBLE_DEVICES=0 python eval_wavernn.py --checkpoint ../../tts-subjective/models/parallel_wavernn_ljspeech_fatchord_ckpt_bs32_ep20k.pt
#3.6803444957733156
#3.213275489807129
#0.9691914456736345

# with pretrained model
# 3.6809009742736816
# 3.2303855085372923
# 0.9692398045562982

python eval_wavernn.py --checkpoint-path ./parallel_wavernn_fatchord_ckpt_v6.pt --n-bits 10
# 2.1663105845451356
# 1.8648222422599792
# 0.9250924429832281

python eval_wavernn.py --checkpoint-path ./parallel_wavernn_fatchord_ckpt_v5.pt --n-bits 8
# 2.6738954830169677
# 2.2965006852149963
# 0.9355676369676148

python eval_wavernn.py --checkpoint-path ./parallel_wavernn_fatchord_ckpt_v4.pt --n-bits 9
# 1.7484594464302063
# 1.507220034599304
# 0.8730921949576181

python eval_wavernn.py --checkpoint-path ./best_parallel_wavernn_fatchord_ckpt_v3.pt
# 1.8433957266807557
# 1.5406161832809449
# 0.8565231596575354

python eval_wavernn.py --checkpoint-path ./parallel_wavernn_fatchord_ckpt_v3.pt
# 2.574930500984192
# 2.1763903856277467
# 0.9365462354715417

python eval_librosa_griffin_lim.py
# 1.375499551296234
# 1.0369670295715332
# 0.839018336562249

python eval_griffin_lim.py
# 1.3719915890693664
# 1.036681363582611
# 0.8339225023896631

python eval_fatchord.py
# 3.462336988449097
# 3.178738441467285
# 0.9579072714495004

# no fade
# 3.472217025756836
# 3.2115252161026
# 0.9579988273908953
