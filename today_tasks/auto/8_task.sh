screen -dmS V_ts_8
screen -x -S V_ts_8 -p 0 -X stuff 'sh today_tasks/V_ts.sh 1.0 8 1 1 BCE 5
'
screen -x -S V_ts_8 -p 0 -X stuff 'sh today_tasks/V_ts.sh 1.0 8 1 1 KLdiv 5
'
screen -x -S V_ts_8 -p 0 -X stuff 'sh today_tasks/V_ts.sh 1.0 8 1 2 BCE 5
'
screen -x -S V_ts_8 -p 0 -X stuff 'sh today_tasks/V_ts.sh 1.0 8 1 2 KLdiv 5
'
screen -x -S V_ts_8 -p 0 -X stuff 'exit
'
