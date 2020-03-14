set -e
for i in `seq 1 1 3`;
do

cmd="python train_baseline.py --dataset_mode=iemocap_10fold --model=early_fusion_multi --gpu_ids=2
--log_dir=./logs --checkpoints_dir=./checkpoints --print_freq=10 
--acoustic_ft_type=IS10 --lexical_ft_type=text --visual_ft_type=denseface
--input_dim_a=1582 --mid_layers_a=256,128
--input_dim_v=342 --hidden_size=128 --embd_size_v=128
--input_dim_l=1024 --embd_size_l=128
--fusion_size=256 --mid_layers_fusion=128,128
--output_dim=4 --modality=AL
--niter=40 --niter_decay=60 --verbose
--batch_size=256 --lr=1e-3 --dropout_rate=0.5 --run_idx=2
--name=ef_AL --suffix=Adnn{mid_layers_a}_Vlstm{hidden_size}_{embd_size_v}_Lcnn{embd_size_l}_fusion{mid_layers_fusion}run{run_idx}
--cvNo=$i"

# --name=

echo "\n-------------------------------------------------------------------------------------"
echo "Execute command: $cmd"
echo "-------------------------------------------------------------------------------------\n"
echo $cmd | sh

done