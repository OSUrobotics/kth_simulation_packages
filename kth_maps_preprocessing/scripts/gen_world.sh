#!/bin/bash -e


for d in $RELATIVE_PATH/../models/*/model.sdf; do
  python buildworld.py $d
done
