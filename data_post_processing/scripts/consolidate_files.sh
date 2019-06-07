#!/bin/bash
trap "exit" INT
RELATIVE_PATH=$( cd "$(dirname "${BASH_SOURCE[0]}")" ; pwd -P )
PACKAGE_PATH="$(dirname "$RELATIVE_PATH")"
shopt -s nullglob

while true; do
  for d in $PACKAGE_PATH/input/*/data.npy; do
    python $PACKAGE_PATH/src/data_consolidation.py $d
    rm -rf "$(dirname $d)"
  done
  sleep 30
done
