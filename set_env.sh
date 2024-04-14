#!/bin/bash
if [ ! -d "logs" ]; then
    mkdir -p "logs"
fi

TKGHOME=$(pwd)
export PYTHONPATH="$TKGHOME:$PYTHONPATH"
export LOG_DIR="$TKGHOME/logs"
export DATA_PATH="$TKGHOME/data"
conda activate regcn

