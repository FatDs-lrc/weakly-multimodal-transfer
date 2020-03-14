set -e
for i in `seq 0 1 4`;
do

cmd="python train_baseline.py --dataset_mode=iemocap_seq --model=didi --gpu_ids=0
--log_dir=./logs --checkpoints_dir=./checkpoints --print_freq=10 
--input_a=34 --input_l=300 --output_dim=4 --dropout_rate=0.3 
--A_ft_type=lld_34 --L_ft_type=glove
--batch_size=128 --lr=1e-3
--name=DiDi --suffix=hidden{hidden_size}
--cvNo=$i"

#  setup_ss_batchsize_128_lr_0.001_tgt_discrete_mark_IS10+bert+decoder+mmd+0.3+negative+valid_ami/
echo "\n-------------------------------------------------------------------------------------"
echo "Execute command: $cmd"
echo "-------------------------------------------------------------------------------------\n"
echo $cmd | sh

done