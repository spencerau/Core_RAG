#!/bin/bash

clear
# need this to work with IS&T, not necessary on MLAT stuff
# ssh -fN -L 10001:localhost:10001 -L 10002:localhost:10002 dgx_cluster

python -m pytest tests/ -v -s