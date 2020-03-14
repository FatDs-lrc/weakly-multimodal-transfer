set -ex
for f in `ssh leo "ls lrc_git/weakly-multimodal-transfer"`; do
    if [ "$f" != "checkpoints" -a "$f" != "cluster_graph" ];then
        if [ "$f" != "logs" -a "$f" != "__pycache__" ];then
            scp -r leo:lrc_git/weakly-multimodal-transfer/$f ./
        fi 
    fi
done
