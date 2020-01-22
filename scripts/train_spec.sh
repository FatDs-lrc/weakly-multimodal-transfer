set -e
cmd="python train_kd.py --dataset_mode=iemocap --model=spectral --gpu_ids=0,1,2,3,4,5,6,7
--log_dir=./logs --checkpoints_dir=./checkpoints --print_freq=10 --batch_size=64
--input_dim=1582 --mid_layers=512,256,128 --output_dim=4 --dropout_rate=0.2 --modality=acoustic
--teacher_path=/data2/ljj/SSER_model/setup_ss_batchsize_128_lr_0.001_tgt_discrete_mark_IS10+bert+decoder+mmd+0.3+negative+valid_ami/
--adjacent_path=graph/assets/adjacent_matrix_knn100_sigma1.npy
--kd_temp=4 --spec_weight=0 --orth_weight=0 --kd_start_epoch=0
--name=TS_acoustic --suffix=Spec{spec_weight}_Orth{orth_weight}_KDtemp{kd_temp}_layers{mid_layers}_cvNo{cvNo}
--cvNo=5"

echo "\n-------------------------------------------------------------------------------------"
echo "Execute command: $cmd"
echo "-------------------------------------------------------------------------------------\n"
echo $cmd | sh