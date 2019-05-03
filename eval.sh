CUDA_VISIBLE_DEVICES=2,3 python train.py checkpoints/clsorderim_moment_lr0.0003/config --checkpoint 11.model --test test1 -t 12 -b 32 -p 200
CUDA_VISIBLE_DEVICES=2,3 python train.py checkpoints/cnn_moment_lr0.001/config --checkpoint 3.model --test test1 -t 12 -b 32 -p 200
# CUDA_VISIBLE_DEVICES=2,3 python train.py checkpoints/convlstm_lr0.0003/config --checkpoint 18.model --test test1 -t 12 -b 32 -p 200
# CUDA_VISIBLE_DEVICES=2,3 python train.py checkpoints/convlstml/config --checkpoint 12.model --test test1 -t 12 -b 32 -p 200

