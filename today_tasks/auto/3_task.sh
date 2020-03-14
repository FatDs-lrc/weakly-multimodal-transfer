screen -dmS V_ts_3
screen -x -S V_ts_3 -p 0 -X stuff 'sh today_tasks/V_ts.sh 1.0 4 0 1 BCE 2
'
screen -x -S V_ts_3 -p 0 -X stuff 'sh today_tasks/V_ts.sh 1.0 4 0 1 KLdiv 2
'
screen -x -S V_ts_3 -p 0 -X stuff 'sh today_tasks/V_ts.sh 1.0 4 0 2 BCE 2
'
screen -x -S V_ts_3 -p 0 -X stuff 'sh today_tasks/V_ts.sh 1.0 4 0 2 KLdiv 2
'
screen -x -S V_ts_3 -p 0 -X stuff 'exit
'
