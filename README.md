# Lifelong Property Price Prediction: A Case Study for the Toronto Real Estate Market
Implementation of [LUCE](https://arxiv.org/abs/2008.05880) paper



### Environment
``` 
conda create -n $ENV_NAME$ python=3.7
conda activate $ENV_NAME$

# CUDA 11.3
pip install torch==1.11.0+cu113 --extra-index-url https://download.pytorch.org/whl/cu113 
# Or, CUDA 10.2 
pip install torch==1.11.0+cu102 --extra-index-url https://download.pytorch.org/whl/cu102 
pip install -r requirements.txt
```

### Train model

### Reference
* Official Implementation: https://github.com/RingBDStack/LUCE
