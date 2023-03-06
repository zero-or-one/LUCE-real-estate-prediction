# Real Estate Price Prediction
Implementation of several real estate prediction papers:
1. PDVM: https://link.springer.com/article/10.1007/s41688-018-0027-0


### Preprocess
```
python preprocess.py
```

### Training the model
Go to the desired model directory \
Specify the parameters in required class in config.py
```
python train_{model_name}.py
```

### Results
| Model | MSE |
|-------|--|
| HA | 0.063 |
| LR | 0.084 |
| SVM | 0.066 |
| GBRT | 0.019 |
| DNN | 21.38 |
| PDVM | 0.238 |


### Reference
