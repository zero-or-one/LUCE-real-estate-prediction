
import os 
import torch


class Logger():
    def __init__(self, result_file_path, model_file_path, other_file_path):
        self.result_file_path = result_file_path
        self.other_file_path = other_file_path
        self.model_file_path = model_file_path
        for output_path in [self.result_file_path, self.model_file_path, self.other_file_path]:
            if not os.path.isdir(output_path):
                os.makedirs(output_path)

    def save_parameters(self, config):
        with open(self.result_file_path+'parameters.txt', 'w') as f:
            f.write('data_path: '+str(config.data_path)+'\n')
            f.write('input_size: '+str(config.input_size)+'\n')
            f.write('layers: '+str(config.layers)+'\n')
            f.write('dropout: '+str(config.dropout)+'\n')
            f.write('epoch: '+str(config.epoch)+'\n')
            f.write('seq_len: '+str(config.seq_len)+'\n')
            f.write('house_size: '+str(config.house_size)+'\n')
            f.write('lr: '+str(config.lr)+'\n')
            f.write('weight_decay: '+str(config.weight_decay)+'\n')
            f.write('device: '+str(config.device)+'\n')

    def save_model(self, model, optimizer, epoch):
        torch.save(model.state_dict(), self.model_file_path+'model_'+str(epoch)+'.pkl')
        torch.save(optimizer.state_dict(), self.model_file_path+'optimizer_'+str(epoch)+'.pkl')

    def log_testing(self, epoch, mse, mae, rmse, y_pre_error, cost_time, w_str):
        with open(self.result_file_path+'loss_error.txt', 'a+') as f:
            f.write("Test MSE: {} MAE:{} RMSE: {} pre_error:{} cost_time:{}\n".format(mse,mae,rmse,y_pre_error,cost_time))
        with open(self.other_file_path + 'valid_RMSE.txt', 'a+') as f:
            f.write("{}\n".format(rmse))
        with open(self.other_file_path + 'pre_error.txt', 'a+') as f:
            f.write("{}\n".format(y_pre_error))
        with open(self.other_file_path+'price_list.csv', 'w') as f:
            f.write('index, list, target, pre, pre-target, list-target\n')
            f.write(w_str)
    
    def log_training(self, epoch, avg_training_loss):
        with open(self.result_file_path+'loss_error.txt', 'a+') as f:
            f.write("Epoch:{}  Training loss:{}\n".format(epoch, avg_training_loss))
        with open(self.other_file_path+'train_loss.txt', 'a+') as f:
            f.write("{}\n".format(avg_training_loss))
    
