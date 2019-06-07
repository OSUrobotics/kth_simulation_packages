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
0510032322
0510034881_A-40
0510040946_A-40
50010541
50041169
50045233
50055639
0510025537_Layout1_PLAN2
0510032196_A_40_1_104
0510032323
0510035341_A_40_1_103
0510040985_A-40
50041170
50045365
50055640
0510025537_Layout1_PLAN3
0510032197_A_40_1_105
0510034689_Layout1
0510035342_A_40_1_101
0510045903_A_40_1_101
50041171
50045366
50055641
0510028106_Layout1
0510032198_A_40_1_106
0510034690_Layout1
0510035343_A_40_1_102
50041172
50055642
0510030937_A_40_1_106
0510034691_Layout1
0510035345_A_40_1_104
50015850_PLAN4
50041173
50045368
50055647
0510030938_A_40_1_102
0510034692_Layout1
0510045906_A_40_1_104
50015850_PLAN5
50041174
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

trap "exit" INT
RELATIVE_PATH=$( cd "$(dirname "${BASH_SOURCE[0]}")" ; pwd -P )
PACKAGE_PATH="$(dirname "$RELATIVE_PATH")"
PACKAGE_PATH="$(dirname "$PACKAGE_PATH")"

for d in "${WORLDS[@]}"; do
  python $PACKAGE_PATH/src/highway_predictions/generate_visits.py $PACKAGE_PATH/compressed_data/$d/ $PACKAGE_PATH/processed_data/$d/
done
