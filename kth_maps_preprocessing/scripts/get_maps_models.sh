#!/bin/bash -e

RELATIVE_PATH="`dirname \"$0\"`"

for d in $RELATIVE_PATH/../source_files/*/*.xml; do
  python model_map_builder.py $d
done
