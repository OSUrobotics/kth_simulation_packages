#!/bin/bash
WORLDS=(0510025536_Layout1_PLAN4
0510032192_A_40_1_203
0510032270_A_40_1_104
0510034879_A-40
0510040942_A-40
50010539
50037764_PLAN2
50045231
50055637
0510025536_Layout1_PLAN5
0510032194_A_40_1_102
0510032271_A_40_1_105
0510034880_A-40
0510040943_A-40
50010540
50037765_PLAN3
50045232
50055638
0510025537_Layout1_PLAN1
0510032194_A_40_1_202
  )

  for i in "${WORLDS[@]}"
  do
     docker run -e "KTH_WORLD=$i" -e "STORAGE_DOMAIN=$STORAGE_DOMAIN" -d kth_simulation
  done
  sleep 2
  start_time="$(date -u +%s)"

  while true
  do
     var=$(docker ps | grep kth_simulation | wc -l)
     if [ $var -eq 0 ]; then
        sudo shutdown -h now;
     fi
     cur_time="$(date -u +%s)"
     elapsed="$(($cur_time-$start_time))"
     if [ $elapsed -gt 260000 ]; then
        sudo shutdown -h now;
     fi
  done
