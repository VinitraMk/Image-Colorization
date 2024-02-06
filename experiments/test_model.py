#script for testing the model

from common.utils import get_exp_params, get_accuracy, get_config, get_model_filename
from torch.utils.data import DataLoader
import torch
from matplotlib import pyplot as plt
from common import colorspaces

class ModelTester:

    def __init__(self, model, te_dataset):
        cfg = get_config()
        self.te_dataset = te_dataset
        self.model = model.cpu()
        self.model.eval()
        self.exp_params = get_exp_params()
        self.te_loader = DataLoader(self.te_dataset,
            batch_size = self.exp_params['train']['batch_size'],
            shuffle = False
        )
        self.output_dir = cfg['output_dir']
        self.X_key = cfg['X_key']
        self.y_key = cfg['y_key']

    def __loss_fn(self, loss_name = 'cross-entropy'):
        if loss_name == 'cross-entropy':
            return torch.nn.CrossEntropyLoss()
        elif loss_name == 'mse':
            return torch.nn.MSELoss()
        elif loss_name == 'l1':
            return torch.nn.L1Loss()
        else:
            raise SystemExit("Error: no valid loss function name passed! Check run.yaml")

    def __plot_predicted_images(self, L, AB_pred, RGB):
        L = L.transpose(1, 3).transpose(1, 2)
        AB_pred = AB_pred.transpose(1, 3).transpose(1, 2)
        pred_LAB = torch.concat((L[:, :, :, :], AB_pred.detach()), dim = 3)
        n = len(self.te_dataset)
        plt.clf()
        plt.figure(figsize=(n,5))
        for i in range(n):
            pos = i + 1
            #Plot true image
            plt.subplot(5,n,pos)
            plt.imshow(RGB[i,:,:,:])
            plt.axis(False)

            pos = i + n + 1
            #Plot L channel
            plt.subplot(5,n,pos)
            plt.imshow(L[i,:,:],cmap="gray")
            plt.axis(False)
            
            pos = i + (2 * n) + 1
            #Plot A channel
            plt.subplot(5,n,pos)
            plt.imshow(AB_pred[i,:,:,0].detach())
            plt.axis(False)

            pos = i + (3 * n) + 1
            #Plot B channel
            plt.subplot(5,n,pos)
            plt.imshow(AB_pred[i,:,:,1].detach())
            plt.axis(False)

            pos = i + (4 * n) + 1 
            #Convert LAB prediction to RGB and plot
            pred_RGB = colorspaces.lab_to_rgb(pred_LAB[i,:,:,:])
            plt.subplot(5,n,pos)
            plt.imshow(pred_RGB[0,:,:,:])
            plt.axis(False)
    
        plt.savefig(f'{self.output_dir}/test_results.png')
        plt.show()
   
    def test_and_plot(self, RGB, model_type = 'best_model', save_model_temporarily = False):
        loss_fn = self.__loss_fn(self.exp_params["train"]["loss"])
        running_loss = 0.0
        acc = 0
        for _, batch in enumerate(self.te_loader):
            op = self.model(batch[self.X_key])
            loss = loss_fn(op, batch[self.y_key])
            running_loss += loss.item()
            acc = get_accuracy(op, batch[self.y_key])
            self.__plot_predicted_images(batch[self.X_key], op, RGB)
        print("\nTest Loss:", running_loss/len(self.te_loader))
        print("Test Accuracy:", acc/len(self.te_loader), "\n")

        