set -e
for i in `seq 1 1 10`;
do

cmd="python train_baseline.py --dataset_mode=iemocap_10fold --model=early_fusion_multi --gpu_ids=0
--log_dir=./logs --checkpoints_dir=./checkpoints --print_freq=10 
--acoustic_ft_type=IS10 --lexical_ft_type=text --visual_ft_type=denseface
--input_dim_a=1582 --mid_layers_a=384,64
--input_dim_v=342 --hidden_size=128 --embd_size_v=128 --embd_method_v=last
--input_dim_l=1024 --embd_size_l=128
--fusion_size=128 --mid_layers_fusion=128
--output_dim=4 --modality=V
--niter=20 --niter_decay=60 --verbose
--batch_size=256 --lr=1e-3 --dropout_rate=0.5 --run_idx=4
--name=V_baseline --suffix=Vlstm_1direc_{hidden_size}_{embd_method_v}_fc{mid_layers_fusion}_bn{bn}_run{run_idx}
--cvNo=$i"

# --name=V_baseline --suffix=Vlstm_1direc_{hidden_size}_{embd_method}_fc{mid_layers_fusion}_bn{bn}_run{run_idx}
# --name=V_baseline --suffix=Vlstm{hidden_size}_fc{fusion_size},{mid_layers_fusion}_bn{bn}_run{run_idx}
echo "\n-------------------------------------------------------------------------------------"
echo "Execute command: $cmd"
echo "-------------------------------------------------------------------------------------\n"
echo $cmd | sh

done