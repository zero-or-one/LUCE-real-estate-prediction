import os
import sys
import time
import pandas as pd
import numpy as np
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from torch.utils.data.dataset import random_split

sys.path.append('../')
from data import KNSHS_Dataset
from model import B_LSTM
from config import DefaultConfig
from utils import seed_everything, score
from logger import Logger


if __name__ == '__main__':
    # Set device and load config
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    seed_everything
    config = DefaultConfig(device)
    ckpt_path = config.ckpt_path
    if not os.path.isdir(ckpt_path):
        os.makedirs(ckpt_path)

    # Load data
    df = pd.read_csv(config.data_path + config.dataset)
    if config.year == 'all':
        pass
    else:
        # we choose only 1 year
        df = df[df['year'] == config.year]
    # remove duplicate houses
    df = df.drop_duplicates(subset=['house'], keep='last')
    lons = df['lon_x'].values
    lats = df['lat_y'].values
    prices = df['price'].values
    df = df.drop(['lon_x', 'lat_y', 'price', 'year', 'house'], axis=1)
    df = df.values.astype(np.float32)

    # Prepare data for training
    dataset = KNSHS_Dataset(df, prices, lons, lats, config.K, config.distance_limit)
    train_size = int(config.train_ratio*len(dataset))
    valid_size = len(dataset) - train_size 
    train_dataset, valid_dataset = random_split(dataset, [train_size, valid_size])
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config.batch_size, shuffle=False)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=config.batch_size, shuffle=False)

    # Define others
    logger = Logger()
    model = B_LSTM(config).to(device)
    loss_fn = torch.nn.L1Loss()
    optimizer = torch.optim.RMSprop(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)  
    #scheduler = ReduceLROnPlateau(optimizer, 'min', patience=30, factor=0.1, min_lr=1e-8)
    K = config.K

    for epoch in range(config.epoch_num):
        start_time = time.time()

        #Train
        model.train()
        avg_loss = 0.
        avg_score = ()
        for i, (x_batch, y_batch) in enumerate(train_loader):
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            optimizer.zero_grad()
            y_pred = model(x_batch)
            #print(y_pred)
            loss = loss_fn(y_pred, y_batch)
            loss.backward()
            optimizer.step()
            avg_loss += loss.item()
            avg_score += score(y_pred.detach().cpu(), y_batch.detach().cpu())
        avg_loss /= len(train_loader)
        avg_score = list(map(lambda x: x/len(train_loader), avg_score)) 
        logger.log_training(avg_loss, avg_score, epoch)

        #Evaluate
        model.eval()
        avg_val_loss = 0.
        avg_val_score = ()
        for i, (x_batch, y_batch) in enumerate(valid_loader):
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            y_pred = model(x_batch)
            avg_val_loss += loss_fn(y_pred, y_batch).item()
            avg_val_score += score(y_pred.detach().cpu(), y_batch.detach().cpu())
        avg_val_loss /= len(valid_loader)
        avg_val_score = list(map(lambda x: x/len(valid_loader), avg_val_score))
        logger.log_validation(avg_val_loss, avg_val_score, epoch)

        #Print output of epoch
        elapsed_time = time.time() - start_time
        #scheduler.step(avg_val_loss)
        if epoch%10 == 0:
            print('Epoch {}/{} \t loss={:.4f} \t mape={:.4f} \t val_loss={:.4f} \t val_mape={:.4f} \t time={:.2f}s'.format(epoch + 1, config.epoch_num, avg_loss, avg_score[2], avg_val_loss, avg_val_score[2], elapsed_time))
            torch.save(model.state_dict(), ckpt_path+'model_'+str(epoch)+'.pt')
            torch.save(optimizer.state_dict(), ckpt_path+'optimizer_'+str(epoch)+'.pt')
