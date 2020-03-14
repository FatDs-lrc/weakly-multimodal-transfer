screen -dmS V_ts_5
screen -x -S V_ts_5 -p 0 -X stuff 'sh today_tasks/V_ts.sh 1.0 4 1 1 BCE 3
'
screen -x -S V_ts_5 -p 0 -X stuff 'sh today_tasks/V_ts.sh 1.0 4 1 1 KLdiv 3
'
screen -x -S V_ts_5 -p 0 -X stuff 'sh today_tasks/V_ts.sh 1.0 4 1 2 BCE 3
'
screen -x -S V_ts_5 -p 0 -X stuff 'sh today_tasks/V_ts.sh 1.0 4 1 2 KLdiv 3
'
screen -x -S V_ts_5 -p 0 -X stuff 'exit
'
