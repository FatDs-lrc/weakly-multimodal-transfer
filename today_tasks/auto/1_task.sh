screen -dmS V_ts_1
screen -x -S V_ts_1 -p 0 -X stuff 'sh today_tasks/V_ts.sh 1.0 2 0.5 1 BCE 0
'
screen -x -S V_ts_1 -p 0 -X stuff 'sh today_tasks/V_ts.sh 1.0 2 0.5 1 KLdiv 0
'
screen -x -S V_ts_1 -p 0 -X stuff 'sh today_tasks/V_ts.sh 1.0 2 0.5 2 BCE 1
'
screen -x -S V_ts_1 -p 0 -X stuff 'sh today_tasks/V_ts.sh 1.0 2 0.5 2 KLdiv 1
'
screen -x -S V_ts_1 -p 0 -X stuff 'exit
'
