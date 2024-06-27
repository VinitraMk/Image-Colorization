#script for testing the model

from common.utils import get_exp_params, get_accuracy, get_config, get_model_filename
from torch.utils.data import DataLoader
import torch
from matplotlib import pyplot as plt
from common import colorspaces
from common.colorspaces import rgb_to_lab
from torchvision import transforms


class ModelTester:

    def __init__(self, model, te_dataset, data_transform = None, result_sample_len = 5):
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
        self.data_transform = data_transform
        self.result_sample_len = result_sample_len

    def __loss_fn(self, loss_name = 'cross-entropy'):
        if loss_name == 'cross-entropy':
            return torch.nn.CrossEntropyLoss()
        elif loss_name == 'mse':
            return torch.nn.MSELoss()
        elif loss_name == 'l1':
            return torch.nn.L1Loss()
        else:
            raise SystemExit("Error: no valid loss function name passed! Check run.yaml")

    def __plot_predicted_images(self, L, AB_pred, RGB, AB):
        L = L.transpose(1, 3).transpose(1, 2)
        if not(torch.is_tensor(AB)):
            AB = torch.from_numpy(AB)
        AB = AB.transpose(1, 3).transpose(1, 2)
        AB_pred = AB_pred.transpose(1, 3).transpose(1, 2)
        pred_LAB = torch.concat((L, AB_pred.detach()), dim = 3)
        n = self.result_sample_len
        plt.clf()
        plt.figure(figsize=(n,7))
        #print(RGB.size(), L.size(), AB_pred.size(), pred_LAB.size())
        #print(AB_pred.min(), AB_pred.max(), AB.min(), AB.max())
        for i in range(self.result_sample_len):
            pos = i + 1
            #Plot true image
            plt.subplot(7,n,pos)
            plt.imshow(RGB[i,:,:,:])
            plt.axis(False)

            pos = i + n + 1
            #Plot L channel
            plt.subplot(7,n,pos)
            #print(L[i].min(), L[i].max())
            plt.imshow(L[i,:,:], cmap='gray')
            plt.axis(False)

            pos = i + (2 * n) + 1
            #Plot true A channel
            plt.subplot(7, n, pos)
            plt.imshow(AB[i,:,:, 0])
            plt.axis(False)

            pos = i + (3 * n) + 1
            #Plot predicted A channel
            plt.subplot(7,n,pos)
            plt.imshow(AB_pred[i,:,:,0].detach())
            plt.axis(False)

            pos = i + (4 * n) + 1
            #Plot true B channel
            plt.subplot(7, n, pos)
            plt.imshow(AB[i,:,:,1])
            plt.axis(False)

            pos = i + (5 * n) + 1
            #Plot predicted B channel
            plt.subplot(7,n,pos)
            plt.imshow(AB_pred[i,:,:,1].detach())
            plt.axis(False)

            pos = i + (6 * n) + 1
            #Convert LAB prediction to RGB and plot
            pred_RGB = colorspaces.lab_to_rgb(pred_LAB[i,:,:,:])
            #print(pred_RGB.min(), pred_RGB.max(), RGB[i].min(), RGB[i].max())
            #pred_RGB = 255.0 * pred_RGB
            #pred_RGB = pred_RGB.type(torch.LongTensor)
            plt.subplot(7,n,pos)
            plt.imshow(pred_RGB[0,:,:,:])
            plt.axis(False)

        plt.savefig(f'{self.output_dir}/test_results.png')
        plt.show()

    def __get_lab_images(self, batch_rgb):
        L = torch.empty((1, 1, self.exp_params['transform']['crop_dim'], self.exp_params['transform']['crop_dim'])) #np.empty((1, 224, 224))
        AB = torch.empty((1, 2, self.exp_params['transform']['crop_dim'], self.exp_params['transform']['crop_dim'])) #np.empty((1, 224, 224, 2))
        inv_transform = transforms.Compose([ transforms.Normalize(mean = [ 0., 0., 0. ],
                                                     std = [ 1/0.229, 1/0.224, 1/0.225 ]),
                                transforms.Normalize(mean = [ -0.485, -0.456, -0.406 ],
                                                     std = [ 1., 1., 1. ]),
                               ])
        for i in range(batch_rgb.shape[0]):
            #rgb_img = batch_rgb[i].transpose(0, 2).transpose(0, 1).float()
            rgb_img = batch_rgb[i].float()
            #print('rgb', rgb_img.min(), rgb_img.max())
            if self.data_transform:
                rgb_img = self.data_transform(rgb_img)
            rgb_img = inv_transform(rgb_img)
            rgb_img = rgb_img.transpose(0, 2).transpose(0, 1)
            lab_img = rgb_to_lab(rgb_img)
            lab_img = lab_img.transpose(3, 1).transpose(3, 2)
            l = lab_img[:, 0, :, :]
            ab = lab_img[:, 1:, :, :]
            L = torch.cat((L, l[:, None, :, :]), axis=0) #np.append(L, l, axis=0)
            AB = torch.cat((AB, ab), axis=0)  #np.append(AB, ab, axis=0)
        L = L[1:]
        AB = AB[1:]
        return L, AB

    def test_and_plot(self, RGB, ABtrue, model_type = 'best_model', save_model_temporarily = False):
        loss_fn = self.__loss_fn(self.exp_params["train"]["loss"])
        running_loss = 0.0
        acc = 0
        for _, batch in enumerate(self.te_loader):
            op = self.model(batch[self.X_key])
            loss = loss_fn(op, batch[self.y_key])
            running_loss += loss.item()
            acc = get_accuracy(op, batch[self.y_key])
            self.__plot_predicted_images(batch[self.X_key], op, RGB, ABtrue)
        print("\nTest Loss:", running_loss/len(self.te_loader))
        print("Test Accuracy:", acc/len(self.te_loader), "\n")

    def test_imagenette_and_plot(self, model_type = 'best_model'):
        loss_fn = self.__loss_fn(self.exp_params["train"]["loss"])
        running_loss = 0.0
        acc = 0

        for batch_idx, batch in enumerate(self.te_loader):
            batch[self.X_key], batch[self.y_key] = self.__get_lab_images(batch['RGB'])
            op = self.model(batch[self.X_key])
            loss = loss_fn(op, batch[self.y_key])
            running_loss += loss.item()
            acc = get_accuracy(op, batch[self.y_key])
            RGB = batch['RGB'].transpose(1, 3).transpose(1, 2)
            if batch_idx == 0:
                self.__plot_predicted_images(batch[self.X_key], op, RGB, batch[self.y_key])
            del RGB
        print("\nTest Loss:", running_loss/len(self.te_loader))
        print("Test Accuracy:", acc/len(self.te_loader), "\n")



