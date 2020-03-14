screen -dmS V_ts_0
screen -x -S V_ts_0 -p 0 -X stuff 'sh today_tasks/V_ts.sh 1.0 2 0 1 BCE 0
'
screen -x -S V_ts_0 -p 0 -X stuff 'sh today_tasks/V_ts.sh 1.0 2 0 1 KLdiv 0
'
screen -x -S V_ts_0 -p 0 -X stuff 'sh today_tasks/V_ts.sh 1.0 2 0 2 BCE 0
'
screen -x -S V_ts_0 -p 0 -X stuff 'sh today_tasks/V_ts.sh 1.0 2 0 2 KLdiv 0
'
screen -x -S V_ts_0 -p 0 -X stuff 'exit
'
