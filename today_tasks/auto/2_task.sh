screen -dmS V_ts_2
screen -x -S V_ts_2 -p 0 -X stuff 'sh today_tasks/V_ts.sh 1.0 2 1 1 BCE 1
'
screen -x -S V_ts_2 -p 0 -X stuff 'sh today_tasks/V_ts.sh 1.0 2 1 1 KLdiv 1
'
screen -x -S V_ts_2 -p 0 -X stuff 'sh today_tasks/V_ts.sh 1.0 2 1 2 BCE 1
'
screen -x -S V_ts_2 -p 0 -X stuff 'sh today_tasks/V_ts.sh 1.0 2 1 2 KLdiv 1
'
screen -x -S V_ts_2 -p 0 -X stuff 'exit
'
