set -e
for i in `seq 0 1 4`;
do
cmd="python train_kd.py --dataset_mode=iemocap --model=spectral --gpu_ids=2,0,1,3
--log_dir=./logs --checkpoints_dir=./checkpoints --print_freq=10 --batch_size=128
--input_dim_a=1582 --mid_layers=1024,128,16 --output_dim=4 --dropout_rate=0.3 --modality=acoustic
--teacher_path=/data2/ljj/SSER_model/ide_setup_ss_batchsize_128_lr_0.001_tgt_discrete_mark_session_de_a_l_v_WAP
--adjacent_path=graph/assets/adjacent_matrix_knn100_sigma1.npy
--kd_temp=20 --spec_weight=10 --orth_weight=1e-3 --center_weight=1e-2 --kd_start_epoch=0 --lr=1e-4
--name=Spec_acoustic --suffix=Spec{spec_weight}_Orth{orth_weight}_Center{center_weight}_KDtemp{kd_temp}_layers{mid_layers}
--cvNo=$i"

echo "\n-------------------------------------------------------------------------------------"
echo "Execute command: $cmd"
echo "-------------------------------------------------------------------------------------\n"
echo $cmd | sh

done