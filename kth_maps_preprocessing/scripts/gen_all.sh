#!/bin/bash

trap "exit" INT

RELATIVE_PATH=$( cd "$(dirname "${BASH_SOURCE[0]}")" ; pwd -P )

for d in $RELATIVE_PATH/../source_files/*/*.xml; do
  python $RELATIVE_PATH/../src/modelMapBuilder.py $d
done

for d in $RELATIVE_PATH/../models/*/model.sdf; do
  python $RELATIVE_PATH/../src/buildworld.py $d
done

for d in $RELATIVE_PATH/../maps/*/; do
  python $RELATIVE_PATH/../src/findGoals.py $d
done
