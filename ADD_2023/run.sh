# training command, based on 8 2080-Ti GPUs
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 nohup python -m torch.distributed.launch --nproc_per_node=8 train_wav2vec_melnet_bdr.py hparams/train_wav2vec_melnet_bdr.yaml --distributed_launch --distributed_backend='nccl' --find_unused_parameters > log/train_wav2vec_melnet_bdr.log &

# inference command
CUDA_VISIBLE_DEVICES=0 nohup python infer_wav2vec_melnet_bdr.py hparams/infer_wav2vec_melnet_bdr.yaml > log/infer_wav2vec_melnet_bdr_ADD2022Test.log &


