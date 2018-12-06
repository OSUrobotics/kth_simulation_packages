#!/bin/bash -e

RELATIVE_PATH=$( cd "$(dirname "${BASH_SOURCE[0]}")" ; pwd -P )
trap "exit" INT

for d in $RELATIVE_PATH/../source_files/*/*.xml; do
  python $RELATIVE_PATH/../src/mapBuilder.py $d
done
