# Lifelong Property Price Prediction
Implementation of [LUCE paper](https://arxiv.org/abs/2008.05880)


### Preprocess
Create adjacency matrix and one hot encoding
```
python preprocess.py --data_path $path_to_csv_dataset --create_adj $wheither_to_produce_adj
```
If montly based data needs to be trained instead of yearly, run
```
python preprocess_montly.py --data_path $path_to_csv_dataset --create_adj $wheither_to_produce_adj
```

### Training the model
Specify the parameters in required class in config.py
```
# train lstm model independently
python train_lstm.py

# train gcn model independently
python train.py --config $gcn_config

# train prelifelong model
python train_prelifelong.py
```

### Results
| Model | MSE | MAE |
|-------|--|--|
| LSTM | 0.008 | 0.261 |
| GCN | 0.125 | 0.237 |
| LUCE (yearly) | 0.007 | 0.035 |
| LUCE (monthly) | 0.010 | 0.051 |


### Things to improve
- [x] Tune hparams
- [x] Make the script to train the model end-to-end
- [x] Creation of adjacency matrix is of O(n^2) complexity. Optimization will save a lot of time 


### Reference
* Official Implementation: https://github.com/RingBDStack/LUCE




