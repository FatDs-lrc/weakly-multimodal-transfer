set -e
kd_weight=$1
temp=$2
ce_weight=$3
run_idx=$4
gpu=$5
for i in `seq 1 1 10`;
do

cmd="python train_kd.py --dataset_mode=iemocap_10fold --model=ts --gpu_ids=$gpu
--log_dir=./logs --checkpoints_dir=./checkpoints --print_freq=4
--acoustic_ft_type=IS10 --lexical_ft_type=text --visual_ft_type=denseface
--input_dim_a=1582 --mid_layers_a=512,256
--input_dim_v=342 --hidden_size_v=128 --embd_size_v=128 --embd_method_v=maxpool
--input_dim_l=1024 --embd_size_l=128
--fusion_size=128 --mid_layers_fusion=128
--output_dim=4 --modality=L
--teacher_path='checkpoints/ef_AVL_Adnn512,256,128_Vlstm128_maxpool_Lcnn128_fusion256,128run1'
--teacher_mmd_layers=None --student_mmd_layers=None
--ce_weight=$ce_weight --kd_weight=$kd_weight --mmd_weight=0 --kd_temp=$temp
--niter=20 --niter_decay=80 --verbose
--batch_size=256 --lr=6e-4 --dropout_rate=0.5 --run_idx=$run_idx
--name=L_ts --suffix=Lcnn{embd_size_l}_fc{mid_layers_fusion}_loss{kd_loss_type}_kd{kd_weight}_temp{kd_temp}_ce{ce_weight}_mmd{mmd_weight}_run{run_idx}
--cvNo=$i"

# --name=ef_AVL --suffix=Adnn{mid_layers_a}_Vlstm{hidden_size}_{embd_method}_Lcnn{embd_size_l}_fusion{mid_layers_fusion}run{run_idx}

echo "\n-------------------------------------------------------------------------------------"
echo "Execute command: $cmd"
echo "-------------------------------------------------------------------------------------\n"
echo $cmd | sh
done