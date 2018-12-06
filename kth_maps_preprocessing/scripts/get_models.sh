#!/bin/bash -e

RELATIVE_PATH="`dirname \"$0\"`"
trap "exit" INT

for d in $RELATIVE_PATH/../source_files/*/*.xml; do
  python ../src/worldBuilder.py $d
done
