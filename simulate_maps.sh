#!/bin/bash
WORLDS=(50041174
50045369
50055648
0510030939_A_40_1_103
0510034693_Layout1
0510045907_A_40_1_105
50041175
50052748
50055649
0510030940_A_40_1_104
50041184
50052749
50056456
50025630
50041185
50052750
50056457
0510030942_A_40_1_101
0510032259_A_40_1_101
0510034841_A-40
  )

domain=$STORAGE_DOMAIN

  for i in "${WORLDS[@]}"
  do
     docker run -e "KTH_WORLD=$i" -e "STORAGE_DOMAIN=$domain" -d kth_simulation
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
