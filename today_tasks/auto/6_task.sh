screen -dmS V_ts_6
screen -x -S V_ts_6 -p 0 -X stuff 'sh today_tasks/V_ts.sh 1.0 8 0 1 BCE 4
'
screen -x -S V_ts_6 -p 0 -X stuff 'sh today_tasks/V_ts.sh 1.0 8 0 1 KLdiv 4
'
screen -x -S V_ts_6 -p 0 -X stuff 'sh today_tasks/V_ts.sh 1.0 8 0 2 BCE 4
'
screen -x -S V_ts_6 -p 0 -X stuff 'sh today_tasks/V_ts.sh 1.0 8 0 2 KLdiv 4
'
screen -x -S V_ts_6 -p 0 -X stuff 'exit
'
