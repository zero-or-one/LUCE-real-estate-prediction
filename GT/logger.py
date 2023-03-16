
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

    def save_parameters(self, MODEL_NAME, params, net_params):
        with open(self.result_file_path+'parameters' + '.txt', 'w') as f:
            f.write("""Model: {}\n\nparams={}\n\nnet_params={}\n\n\nTotal Parameters: \
            {}\n\n""".format(MODEL_NAME, params, net_params, net_params['total_param']))
            
    def save_model(self, model, epoch):
        torch.save(model.state_dict(), self.model_file_path+'model_'+str(epoch)+'.pkl')
        #torch.save(optimizer.state_dict(), self.model_file_path+'optimizer_'+str(epoch)+'.pkl')

    def log_testing(self, epoch, mse, mae, rmse, cost_time):
        with open(self.result_file_path+'loss_error.txt', 'a+') as f:
            f.write("Test MSE: {} MAE:{} RMSE: {} cost_time:{}\n".format(mse,mae,rmse,cost_time))
        with open(self.other_file_path + 'valid_RMSE.txt', 'a+') as f:
            f.write("{}\n".format(rmse))
    
    def log_training(self, epoch, avg_training_loss):
        with open(self.result_file_path+'loss_error.txt', 'a+') as f:
            f.write("Epoch:{}  Training loss:{}\n".format(epoch, avg_training_loss))
        with open(self.other_file_path+'train_loss.txt', 'a+') as f:
            f.write("{}\n".format(avg_training_loss))
    
