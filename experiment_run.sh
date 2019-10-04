#!/bin/bash

source activate DL_GPU_37
cd /home/divl212/ForGit/CNNIQA
for ((i=0; i<100;i++)); do
    python main.py --exp_id $i
done;
source deactivate