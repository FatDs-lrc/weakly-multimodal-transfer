set -e
for i in `seq 1 1 10`;
do

cmd="python train_baseline.py --dataset_mode=iemocap_10fold --model=baseline --gpu_ids=3
--log_dir=./logs --checkpoints_dir=./checkpoints --print_freq=10 
--input_dim_a=1582 --mid_layers=384,64,16 --output_dim=4 --dropout_rate=0.5 --modality=acoustic
--input_dim_l=1024 --hidden_size=256 --fc1_size=128
--fusion_size=128 --modality=acoustic
--acoustic_ft_type=IS10 --lexical_ft_type=text --niter=40 --niter_decay=60
--batch_size=256 --lr=1e-3 --bn=False --run_idx=2

--name=TS_10fold_baseline_nobn --suffix={mid_layers}_bs{batch_size}_dp{dropout_rate}_run{run_idx}
--cvNo=$i"

echo "\n-------------------------------------------------------------------------------------"
echo "Execute command: $cmd"
echo "-------------------------------------------------------------------------------------\n"
echo $cmd | sh

done