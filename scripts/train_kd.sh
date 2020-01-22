set -e
cmd="nohup python train_kd.py --dataset_mode=iemocap --model=teacher_student --gpu_ids=0,1,2,3,4,5,6,7
--log_dir=./logs --checkpoints_dir=./checkpoints --print_freq=10
--input_dim=1582 --mid_layers=512,256,128 --output_dim=4 --dropout_rate=0.3 --modality=acoustic
--teacher_path=/data2/ljj/SSER_model/setup_ss_batchsize_128_lr_0.001_tgt_discrete_mark_IS10+bert+decoder+mmd+0.3+negative+valid_ami/
--kd_weight=0.5 --kd_temp=4 --name=TS_acoustic --suffix=KDweight{kd_weight}_KDtemp{kd_temp}_layers{mid_layers}_cvNo{cvNo}
--cvNo=5"

echo "\n-------------------------------------------------------------------------------------"
echo "Execute command: $cmd"
echo "-------------------------------------------------------------------------------------\n"
echo $cmd | sh