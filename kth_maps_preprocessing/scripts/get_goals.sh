#!/bin/bash
trap "exit" INT

RELATIVE_PATH=$( cd "$(dirname "${BASH_SOURCE[0]}")" ; pwd -P )
PACKAGE_PATH="$(dirname "$RELATIVE_PATH")"
for d in $PACKAGE_PATH/maps/*/; do
  count=7
  while [ "$count" -gt 3 ]
  do
    count="$(ps aux --no-heading | grep findGoals | wc -l)"
  done
  echo $count
  python $PACKAGE_PATH/src/findGoals.py $d &
done
