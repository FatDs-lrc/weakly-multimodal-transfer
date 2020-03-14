set -e

cmd="python test.py --dataset_mode=iemocap --log_dir=./logs 
--checkpoints_dir=./checkpoints  --model=early_fusion_multi --method=mean
--name=miss_ts_miss2_Adnn512,256,128,256,128_lossBCE_kd1.0_temp2.0_ce0.5_mmd0.1_run2"
# TS_acoustic_Spec300.0_Orth0.002_Center0.05_KDtemp4.0_layers512,256,128

echo "\n-------------------------------------------------------------------------------------"
echo "Execute command: $cmd"
echo "-------------------------------------------------------------------------------------\n"
echo $cmd | sh
