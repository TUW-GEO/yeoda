#!/bin/bash

if [  "$1" = "notebook" ]; then
  if [ ! -d "/opt/notebooks" ]; then
    echo "/opt/notebooks folder does not exist. You need to mount one using -v path/to/local/folder:/opt/notebooks for the jupyter server to store its notebook files to."
    exit 1
  fi
  /bin/bash -c "/opt/conda/envs/yeoda/bin/jupyter notebook --notebook-dir=/opt/notebooks --ip='*' --port=8888 --no-browser --allow-root"
else
  /bin/bash
fi