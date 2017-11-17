#!/bin/bash
wget -P ./model https://github.com/jacky-guo/ML2017FALL/releases/download/0.0.0/best_model.h5
python hw3_test.py $1 $2