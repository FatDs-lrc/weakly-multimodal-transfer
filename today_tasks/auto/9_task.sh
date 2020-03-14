screen -dmS A_ts_9
screen -x -S A_ts_9 -p 0 -X stuff 'sh today_tasks/A_ts.sh 0.5 8 1.0 2 5
'
screen -x -S A_ts_9 -p 0 -X stuff 'sh today_tasks/A_ts.sh 0.5 16 1.0 1 5
'
screen -x -S A_ts_9 -p 0 -X stuff 'sh today_tasks/A_ts.sh 0.5 16 1.0 2 5
'
screen -x -S A_ts_9 -p 0 -X stuff 'exit
'
