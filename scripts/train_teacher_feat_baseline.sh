set -e
for i in `seq 1 1 10`;
do

cmd="python train_baseline.py --dataset_mode=iemocap_10fold --model=teacher_test --gpu_ids=1
--log_dir=./logs --checkpoints_dir=./checkpoints --print_freq=10 
--output_dim=4 --dropout_rate=0.5 --modality=A --mid_layers=128,128
--acoustic_ft_type=IS10 --lexical_ft_type=text --niter=10 --niter_decay=20
--teacher_path='checkpoints/ef_AVL_Adnn512,256,128_Vlstm128_maxpool_Lcnn128_fusion256,128run1'
--batch_size=256 --lr=1e-3 --run_idx=2

--name=Teacher_feat_baseLine --suffix={modality}_fc{mid_layers}_run{run_idx}
--cvNo=$i"

echo "\n-------------------------------------------------------------------------------------"
echo "Execute command: $cmd"
echo "-------------------------------------------------------------------------------------\n"
echo $cmd | sh

done