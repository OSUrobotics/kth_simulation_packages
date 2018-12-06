#!/bin/bash -e

RELATIVE_PATH="`dirname \"$0\"`"
trap "exit" INT

for d in $RELATIVE_PATH/../models/*/model.sdf; do
  python $RELATIVE_PATH/../src/buildworld.py $d
done
