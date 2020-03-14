screen -dmS V_ts_7
screen -x -S V_ts_7 -p 0 -X stuff 'sh today_tasks/V_ts.sh 1.0 8 0.5 1 BCE 4
'
screen -x -S V_ts_7 -p 0 -X stuff 'sh today_tasks/V_ts.sh 1.0 8 0.5 1 KLdiv 4
'
screen -x -S V_ts_7 -p 0 -X stuff 'sh today_tasks/V_ts.sh 1.0 8 0.5 2 BCE 5
'
screen -x -S V_ts_7 -p 0 -X stuff 'sh today_tasks/V_ts.sh 1.0 8 0.5 2 KLdiv 5
'
screen -x -S V_ts_7 -p 0 -X stuff 'exit
'
