#!/bin/sh
# Run the LUCE real estate prediction experiment
python3 preprocess_monthly.py --create_adj 0;
python3 train_prelifelong.py