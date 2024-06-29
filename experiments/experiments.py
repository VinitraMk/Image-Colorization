from common.utils import get_exp_params
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
import os
from random import shuffle
from torchvision import transforms

from common.utils import get_img_accuracy, get_config, save_experiment_output, save_experiment_chkpt, load_modelpt, save_model_helpers, get_mssd
from models.unet import UNet
from models.custom_models import get_model
from common.colorspaces import rgb_to_lab
from tqdm import tqdm
import warnings
import cv2

warnings.filterwarnings("ignore")

class Experiment:

    def __get_optimizer(self, model, model_params, optimizer_name = 'Adam'):
        if optimizer_name == 'Adam':
            return torch.optim.Adam(model.parameters(), lr = model_params['lr'], weight_decay = model_params['weight_decay'], amsgrad = model_params['amsgrad'])
        elif optimizer_name == 'SGD':
            return torch.optim.SGD(model.parameters(), lr = model_params['lr'], weight_decay = model_params['weight_decay'], momentum = model_params['momentum'], nesterov= True)
        else:
            raise SystemExit("Error: no valid optimizer name passed! Check run.yaml file")


    def __init__(self, model_name, ftr_dataset, transforms = None, data_type = 'custom'):
        self.exp_params = get_exp_params()
        self.model_name = model_name
        self.ftr_dataset = ftr_dataset
        cfg = get_config()
        self.X_key = cfg['X_key']
        self.y_key = cfg['y_key']
        self.root_dir = cfg["root_dir"]
        self.device = 'cuda' if cfg['use_gpu'] else 'cpu'
        self.all_folds_res = {}
        self.data_transform = transforms
        self.data_type = data_type

    def __loss_fn(self, loss_name = 'cross-entropy'):
        if loss_name == 'cross-entropy':
            return torch.nn.CrossEntropyLoss()
        elif loss_name == 'mse':
            return torch.nn.MSELoss()
        elif loss_name == 'l1':
            return torch.nn.L1Loss()
        else:
            raise SystemExit("Error: no valid loss function name passed! Check run.yaml")
    
    def __get_lab_images(self, batch_rgb):
        L = torch.empty((1, 1, 224, 224)) #np.empty((1, 224, 224))
        AB = torch.empty((1, 2, 224, 224)) #np.empty((1, 224, 224, 2))
        inv_transform = transforms.Compose([ transforms.Normalize(mean = [ 0., 0., 0. ],
                                                     std = [ 1/0.229, 1/0.224, 1/0.225 ]),
                                transforms.Normalize(mean = [ -0.485, -0.456, -0.406 ],
                                                     std = [ 1., 1., 1. ]),
                               ])
        for i in range(batch_rgb.shape[0]):
            #rgb_img = batch_rgb[i].transpose(0, 2).transpose(0, 1).float()
            rgb_img = batch_rgb[i].float()
            if self.data_transform:
                #print('b4 transform', rgb_img.size())
                rgb_img = self.data_transform(rgb_img)
                #print('after transform', rgb_img.size())
            rgb_img = inv_transform(rgb_img) 
            rgb_img = rgb_img.transpose(0, 2).transpose(0, 1)
            #rgb_img = rgb_img[None, :, :, :]
            #print('after transpose', rgb_img.size())
            lab_img = rgb_to_lab(rgb_img)
            #rgb_np = rgb_img.numpy()
            #lab_np = lab_img.numpy()
            #print('after lab', lab_img.size())
            lab_img = lab_img.transpose(3, 1).transpose(3, 2)
            #print('lab img sz', lab_img.size())
            l = lab_img[:, 0, :, :]
            ab = lab_img[:, 1:, :, :]
            #print(l.size(), ab.size())
            L = torch.cat((L, l[:, None, :, :]), axis=0) #np.append(L, l, axis=0)
            AB = torch.cat((AB, ab), axis=0)  #np.append(AB, ab, axis=0)
        L = L[1:]
        AB = AB[1:]
        return L, AB
            
    def __conduct_training(self, model, fold_idx, epoch_index,
                           train_loader, val_loader,
                           train_len, val_len,
                           trlosshistory = [], vallosshistory = [], valerrhistory = []):
        loss_fn = self.__loss_fn(self.exp_params['train']['loss'])
        num_epochs = self.exp_params['train']['num_epochs']
        epoch_ivl = self.exp_params['train']['epoch_interval']
        tr_loss = 0.0
        val_loss = 0.0
        model_info = {}
        epoch_arr = list(range(epoch_index, num_epochs))
        disable_tqdm_log = True
        for i in epoch_arr:
            if ((i+1) % epoch_ivl == 0) or i == 0:
                print(f'\tRunning Epoch {i+1}')
                disable_tqdm_log = False
            model.train()
            tr_loss = 0.0
            for batch_idx, batch in enumerate(tqdm(train_loader, desc = '\t\tRunning through training set', position = 0, leave = True, disable = disable_tqdm_log)):
                self.optimizer.zero_grad()
                #print('batch sz', batch['RGB'].size())
                if self.data_type == 'imagenette':
                    batch[self.X_key], batch[self.y_key] = self.__get_lab_images(batch['RGB'])
                #print('before data transform: ', batch[self.X_key].shape, batch[self.y_key].shape) 
                batch[self.X_key] = batch[self.X_key].float().to(self.device)
                batch[self.y_key] = batch[self.y_key].float().to(self.device)
                op = model(batch[self.X_key])
                loss = loss_fn(op, batch[self.y_key])
                loss.backward()
                self.optimizer.step()
                tr_loss += (loss.item() * batch[self.X_key].size()[0])
                #torch.cuda.empty_cache()
            tr_loss /= train_len
            trlosshistory.append(tr_loss)

            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for batch_idx, batch in enumerate(tqdm(val_loader, desc = '\t\tRunning through validation set', position = 0, leave = True, disable=disable_tqdm_log)):
                    if self.data_type == 'imagenette':
                        batch[self.X_key], batch[self.y_key] = self.__get_lab_images(batch['RGB'])
                    batch[self.X_key] = batch[self.X_key].float().to(self.device)
                    batch[self.y_key] = batch[self.y_key].float().to(self.device)
                    lop = model(batch[self.X_key])
                    loss = loss_fn(lop, batch[self.y_key])
                    val_loss += (loss.item() * batch[self.y_key].size()[0])
                    #torch.cuda.empty_cache()
            val_loss /= val_len
            vallosshistory.append(val_loss)
            if ((i+1) % epoch_ivl == 0) or i == 0:
                print(f'\tEpoch {i+1} Training Loss: {tr_loss}')
                print(f"\tEpoch {i+1} Validation Loss: {val_loss}")

            model_info = {
                'valloss': val_loss,
                'trloss': tr_loss,
                'trlosshistory': torch.tensor(trlosshistory),
                'vallosshistory': torch.tensor(vallosshistory),
                'fold': fold_idx,
                'epoch': i
            }
            self.save_model_checkpoint(model.state_dict(), self.optimizer.state_dict(),
            self.all_folds_res, model_info, False)
            disable_tqdm_log = True

        model_info = {
            'valloss': val_loss,
            'trloss': tr_loss,
            'trlosshistory': torch.tensor(trlosshistory),
            'vallosshistory': torch.tensor(vallosshistory),
            'fold': fold_idx,
            'epoch': -1
        }
        self.all_folds_res[fold_idx] = model_info
        self.save_model_checkpoint(model.state_dict(), self.optimizer.state_dict(),
        self.all_folds_res, model_info, False)
        return model, model_info
    
    def __get_experiment_chkpt(self, model):
        mpath = os.path.join(self.root_dir, "models/checkpoints/current_model.pt")
        if os.path.exists(mpath):
            print("Loading saved model")
            saved_model = load_modelpt(mpath)
            model_dict = saved_model["model_state"]
            model.load_state_dict(model_dict)
            self.all_folds_res = saved_model["model_history"]
            self.optimizer = self.__get_optimizer(model, self.exp_params['model'], self.exp_params['model']['optimizer'])
            ops = saved_model['optimizer_state']
            self.optimizer.load_state_dict(ops)
            return model, saved_model["last_state"]
        else:
            if self.exp_params['model']['build_on_pretrained']:
                # do something
                print("Loading the given pretrained model")
                mpath = os.path.join(self.root_dir, self.exp_params['model']['pretrained_filename'])
                saved_model = load_modelpt(mpath)
                model_dict = saved_model["model_state"]
                model.load_state_dict(model_dict)
                self.all_folds_res = saved_model["model_history"]
                self.optimizer = self.__get_optimizer(model, self.exp_params['model'], self.exp_params['model']['optimizer'])
            else:
                self.optimizer = self.__get_optimizer(model, self.exp_params['model'], self.exp_params['model']['optimizer'])
            return model, None

    def train(self):
        train_loader = {}
        val_loader = {}

        if self.exp_params['train']['val_split_method'] == 'fixed-split':
            print("Running straight split")
            model = get_model(self.model_name)
            model = model.to(self.device)
            model, ls = self.__get_experiment_chkpt(model)
            epoch_index = 0 if ls == None else ls['epoch'] + 1
            vp = self.exp_params['train']['val_percentage'] / 100
            fl = len(self.ftr_dataset)
            vlen = int(vp * fl)
            fr = list(range(fl))
            shuffle(fr)
            val_idxs = fr[:vlen]
            tr_idxs = fr[vlen:]
            train_dataset = Subset(self.ftr_dataset, tr_idxs)
            val_dataset = Subset(self.ftr_dataset, val_idxs)
            tr_len = len(tr_idxs)
            val_len = len(val_idxs)
            train_loader = DataLoader(train_dataset,
                batch_size = self.exp_params['train']['batch_size'],
                shuffle = self.exp_params['train']['shuffle_data']
            )
            val_loader = DataLoader(val_dataset,
                batch_size = self.exp_params['train']['batch_size'],
                shuffle = self.exp_params['train']['shuffle_data']
            )
            if ls != None:
                model, model_info = self.__conduct_training(model, -1, epoch_index,
                    train_loader, val_loader,
                    tr_len, val_len,
                    ls['trlosshistory'].tolist(), ls['vallosshistory'].tolist())
            else:
                model, model_info = self.__conduct_training(model, -1, epoch_index,
                    train_loader, val_loader, tr_len, val_len)
            del model_info['fold']
            del model_info['epoch']
            self.save_model_checkpoint(model.state_dict(), self.optimizer.state_dict(), self.all_folds_res, model_info, True)
            torch.cuda.empty_cache()
            return {}
        else:
            raise SystemExit("Error: no valid split method passed! Check run.yaml")

    def save_model_checkpoint(self, model_state, optimizer_state, model_history, chkpt_info,
    is_final = False):
        if is_final:
            save_experiment_output(model_state, chkpt_info, self.exp_params,
                True)
            save_model_helpers(model_history, optimizer_state, '', True)
            os.remove(os.path.join(self.root_dir, "models/checkpoints/current_model.pt"))
        else:
            save_experiment_chkpt(model_state, optimizer_state, chkpt_info, model_history)



