#!/bin/bash

RELATIVE_PATH=$( cd "$(dirname "${BASH_SOURCE[0]}")" ; pwd -P )
trap "exit" INT
for d in $RELATIVE_PATH/../maps/*/; do
  python $RELATIVE_PATH/../src/findGoals.py $d
done
