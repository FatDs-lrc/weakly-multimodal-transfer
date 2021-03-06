set -e
for i in `seq 1 1 10`;
do

cmd="python train_kd.py --dataset_mode=iemocap_10fold --model=ts --gpu_ids=2
--log_dir=./logs --checkpoints_dir=./checkpoints --print_freq=4
--acoustic_ft_type=IS10 --lexical_ft_type=text --visual_ft_type=denseface
--input_dim_a=1582 --mid_layers_a=512,256
--input_dim_v=342 --hidden_size_v=128 --embd_size_v=128 --embd_method=attention
--input_dim_l=1024 --embd_size_l=128
--fusion_size=256 --mid_layers_fusion=128
--output_dim=4 --modality=A
--teacher_path='checkpoints/ef_AVL_Adnn512,256,128_Vlstm128_maxpool_Lcnn128_fusion256,128run1'
--teacher_mmd_layers=None --student_mmd_layers=None
--ce_weight=1 --kd_weight=1 --mmd_weight=0 --kd_temp=4
--niter=40 --niter_decay=60 --verbose
--batch_size=256 --lr=1e-3 --dropout_rate=0.5 --run_idx=2
--name=A_ts_decay --suffix=Adnn{mid_layers_a},{mid_layers_fusion}_kd{kd_weight}_temp{kd_temp}_ce{ce_weight}_mmd{mmd_weight}_run{run_idx}
--cvNo=$i"

# --name=ef_AVL --suffix=Adnn{mid_layers_a}_Vlstm{hidden_size}_{embd_method}_Lcnn{embd_size_l}_fusion{mid_layers_fusion}run{run_idx}

echo "\n-------------------------------------------------------------------------------------"
echo "Execute command: $cmd"
echo "-------------------------------------------------------------------------------------\n"
echo $cmd | sh

done