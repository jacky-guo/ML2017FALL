#!/bin/bash
wget -P ./ https://www.dropbox.com/s/sw4h8ggd332t8a4/model.h5?dl=1 -O model.h5
python3 test.py $1 $2