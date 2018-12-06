#!/bin/bash -e

RELATIVE_PATH="`dirname \"$0\"`"

for d in $RELATIVE_PATH/../source_files/*/*.xml; do
  python mapBuilder.py $d
done
