#!/bin/bash -e

RELATIVE_PATH=$( cd "$(dirname "${BASH_SOURCE[0]}")" ; pwd -P )
trap "exit" INT

for d in $RELATIVE_PATH/../models/*/model.sdf; do
  python $RELATIVE_PATH/../src/buildworld.py $d
done
