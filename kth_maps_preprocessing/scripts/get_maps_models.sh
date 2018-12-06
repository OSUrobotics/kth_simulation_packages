#!/bin/bash -e

RELATIVE_PATH="`dirname \"$0\"`"

for d in /home/$USER/workspace/kth_maps/source_files/*/*.xml; do
  python model_map_builder.py $d
done
