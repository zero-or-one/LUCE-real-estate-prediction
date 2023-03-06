import torch



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--config" , type=str, default='DefaultConfig')
    parser.add_argument("--cuda", type=bool, default=True)
    parser.add_argument("--visible_devices", type=str, default='0')       
    parser.add_argument("--seed", type=int, default=13)
    args = parser.parse_args()

    torch.cuda.manual_seed(args.seed)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.visible_devices
    device = "cuda" if torch.cuda.is_available() else "cpu"
    config = getattr(config, args.config)(device)

    # Prepare data
    df = pd.read_csv(config.data_path + config.dataset, index_col=False)
    idxs = range(len(df))
    prices = df['price'].values
    # split the data into train and test, test gets last 10% of data
    train_df = df.iloc[:int(len(df) * 0.9)]
    test_df = df.iloc[int(len(df) * 0.9):]
    train_idxs = idxs[:int(len(df) * 0.9)]
    test_idxs = idxs[int(len(df) * 0.9):]
    train_prices = prices[:int(len(df) * 0.9)]
    test_prices = prices[int(len(df) * 0.9):]
    train_dataset = Dataset(train_df, train_prices, train_idxs)
    test_dataset = Dataset(test_df, test_prices, test_idxs)
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)

    # Load adjacencies
    adj_house = np.load(config.data_path + 'adj_house.npy')
    adj_house = torch.from_numpy(adj).float().to(device)
    adj_geo = np.load(config.data_path + 'adj_geo.npy')
    adj_geo = torch.from_numpy(adj).float().to(device)

    # Define others
    logger = Logger()
    model = MugRep(config).to(device)
    loss_fn = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)  

    for epoch in range(config.epoch_num):
        start_time = time.time()
        #Train
        model.train()
        avg_loss = 0.
        avg_score = ()
        for i, (x_batch, y_batch, idxs) in enumerate(train_loader):
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            optimizer.zero_grad()
            y_pred = model(x_batch)
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


