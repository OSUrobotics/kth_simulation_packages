#!/bin/bash -e

RELATIVE_PATH=$( cd "$(dirname "${BASH_SOURCE[0]}")" ; pwd -P )
trap "exit" INT
program=$RELATIVE_PATH/../src/model_map_builder.py
for d in $RELATIVE_PATH/../source_files/*/*.xml; do
  python $RELATIVE_PATH/../src/modelMapBuilder.py $d
done
