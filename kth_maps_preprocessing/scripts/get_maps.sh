#!/bin/bash -e

RELATIVE_PATH="`dirname \"$0\"`"

for d in /home/$USER/workspace/kth_maps/source_files/*/*.xml; do
  python mapBuilder.py $d
done
