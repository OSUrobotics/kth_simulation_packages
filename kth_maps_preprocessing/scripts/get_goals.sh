#!/bin/bash

RELATIVE_PATH="`dirname \"$0\"`"
trap "exit" INT
for d in $RELATIVE_PATH/../maps/*/; do
  python ../src/findGoals.py $d
done
