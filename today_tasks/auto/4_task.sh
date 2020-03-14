screen -dmS V_ts_4
screen -x -S V_ts_4 -p 0 -X stuff 'sh today_tasks/V_ts.sh 1.0 4 0.5 1 BCE 2
'
screen -x -S V_ts_4 -p 0 -X stuff 'sh today_tasks/V_ts.sh 1.0 4 0.5 1 KLdiv 2
'
screen -x -S V_ts_4 -p 0 -X stuff 'sh today_tasks/V_ts.sh 1.0 4 0.5 2 BCE 3
'
screen -x -S V_ts_4 -p 0 -X stuff 'sh today_tasks/V_ts.sh 1.0 4 0.5 2 KLdiv 3
'
screen -x -S V_ts_4 -p 0 -X stuff 'exit
'
