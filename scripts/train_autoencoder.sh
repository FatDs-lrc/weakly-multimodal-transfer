set -e
cmd="python train_autoencoder.py --dataset_mode=mix --model=autoencoder --gpu_ids=0,5,6,7 --log_dir=./logs
--checkpoints_dir=./checkpoints --name=autoencoder_lstm_visual --modality=visual --print_freq=10 
--hidden_size=256 --embedding_size=128 --input_size=342"
echo "\n-------------------------------------------------------------------------------------"
echo "Execute command: $cmd"
echo "-------------------------------------------------------------------------------------\n"
echo $cmd | sh