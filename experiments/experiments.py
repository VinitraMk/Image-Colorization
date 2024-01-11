from common.utils import get_exp_params
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from common.utils import get_accuracy, save_model_chkpt, get_config, save_experiment_output, load_model, save_experiment_chkpt, load_modelpt
from models.unet import UNet
from models.custom_models import get_model
import os

class Experiment:
    
    def __get_optimizer(self, model, model_params, optimizer_name = 'Adam'):
        if optimizer_name == 'Adam':
            return torch.optim.Adam(model.parameters(), lr = model_params['lr'], weight_decay = model_params['weight_decay'], amsgrad = model_params['amsgrad'])
        elif optimizer_name == 'SGD':
            return torch.optim.SGD(model.parameters(), lr = model_params['lr'], weight_decay = model_params['weight_decay'], momentum = model_params['momentum'], nesterov= True)
        else:
            raise SystemExit("Error: no valid optimizer name passed! Check run.yaml file")

    
    def __init__(self, model_name, ftr_dataset):
        self.exp_params = get_exp_params()
        self.model_name = model_name
        self.optimizer = self.__get_optimizer(self.model, self.exp_params['model'], self.exp_params['model']['optimizer'])
        self.ftr_dataset = ftr_dataset
        cfg = get_config()
        self.X_key = cfg['X_key']
        self.y_key = cfg['y_key']
        self.root_dir = cfg["root_dir"]
        self.device = 'cuda' if cfg['use_gpu'] else 'cpu'
        self.all_folds_res = {}
        
    def __loss_fn(self, loss_name = 'cross-entropy'):
        if loss_name == 'cross-entropy':
            return torch.nn.CrossEntropyLoss()
        elif loss_name == 'mse':
            return torch.nn.MSELoss()
        else:
            raise SystemExit("Error: no valid loss function name passed! Check run.yaml")

    def __conduct_training(self, model, fold_idx, epoch_index, train_loader, val_loader):
        loss_fn = self.__loss_fn()
        tr_batch_num = len(train_loader)
        val_batch_num = len(val_loader)
        num_epochs = self.exp_params['train']['num_epochs']
        epoch_ivl = self.exp_params['train']['epoch_interval']
        batch_ivl = self.exp_params['train']['batch_interval']
        trlosshistory = []
        vallosshistory = []
        valacchistory = []
        tr_loss = 0.0
        val_loss = 0.0
        val_acc = 0.0
        model = model.to(self.device).float()
        model_info = {}
        epoch_arr = list(range(epoch_index, num_epochs))
        for i in epoch_arr:
            print(f'\tRunning Epoch {i}')
            model.train()
            tr_loss = 0.0
            print(f'\t\tRunning through training dataset')
            for batch_idx, batch in enumerate(train_loader):
                self.optimizer.zero_grad()
                batch[self.X_key] = batch[self.X_key].float().to(self.device)
                batch[self.y_key] = batch[self.y_key].to(self.device)
                op = model(batch[self.X_key])
                loss = loss_fn(op, batch[self.y_key])
                loss.backward()
                self.optimizer.step()
                tr_loss += loss.item()
                if (batch_idx + 1) % batch_ivl == 0:
                    print(f'\t\tBatch {batch_idx + 1} Loss: {tr_loss / (batch_idx + 1)}')
            tr_loss /= tr_batch_num
            trlosshistory.append(tr_loss)

            print('\t\tRunning through validation set')
            model.eval()
            val_loss = 0.0
            val_acc = 0.0
            with torch.no_grad():
                for batch_idx, batch in enumerate(val_loader):
                    batch[self.X_key] = batch[self.X_key].float().to(self.device)
                    batch[self.y_key] = batch[self.y_key].to(self.device)
                    lop = model(batch[self.X_key])
                    loss = loss_fn(lop, batch[self.y_key])
                    lop_lbls = torch.argmax(lop, 1)
                    val_loss += loss.item()
                    val_acc += get_accuracy(lop_lbls, batch[self.y_key])
    
                    if (batch_idx + 1) % batch_ivl == 0:
                        print(f'\t\tBatch {batch_idx + 1} Last Model Loss: {val_loss / (batch_idx + 1)}')
                        print(f'\t\tBatch {batch_idx + 1} Best Model Loss: {val_loss / (batch_idx + 1)}')
            val_loss /= val_batch_num
            val_acc /= val_batch_num
            vallosshistory.append(val_loss)
            valacchistory.append(val_acc)
            if (i+1) % epoch_ivl == 0:
                print(f'Epoch {i} Training Loss: {tr_loss}')
                print(f"Epoch {i} Validation Loss: {val_loss}")
                print(f"Epoch {i} Validation Accuracy: {val_acc}\n")
            model_info = {
                'valloss': val_loss,
                'valacc': val_acc,
                'trloss': tr_loss,
                'trlosshistory': torch.tensor(trlosshistory),
                'vallosshistory': torch.tensor(vallosshistory),
                'valacchistory': torch.tensor(valacchistory),
                'fold': fold_idx,
                'epoch': i
            }
            self.save_model_checkpoint(model.state_dict(), self.optimizer.state_dict(),
            model_info, self.all_folds_res, 'last_state', True)

        model_info = {
            'valloss': val_loss,
            'valacc': val_acc,
            'trloss': tr_loss,
            'trlosshistory': torch.tensor(trlosshistory),
            'vallosshistory': torch.tensor(vallosshistory),
            'valacchistory': torch.tensor(valacchistory),
            'fold': fold_idx,
            'epoch': -1
        }
        self.all_folds_res[fold_idx] = model_info
        self.save_model_checkpoint(model.state_dict(), self.optimizer.state_dict(),
        model_info, self.all_folds_res, 'last_state', True)
        return model, model_info

    

    def __get_experiment_chkpt(self, model):
        mpath = os.path.join(self.root_dir, "models/checkpoints/current_model.pt")
        if os.path.exists(mpath):
            saved_model = load_modelpt(mpath)
            model_state = model.state_dict()
            model_dict = saved_model["model_state"]
            for key in model_dict:
                model_state[key] = model_dict[key]
            self.all_folds_res = saved_model["model_history"]
            self.optimizer.load_state_dict(saved_model['optimizer_state'])
            return model, saved_model["last_state"], saved_model["best_state"]
        else:
            return model, None, None

    def train(self):
        train_loader = {}
        val_loader = {}
        model = get_model(self.model_name)
        model, ls, bs = self.__get_experiment_chkpt(model)

        if self.exp_params['train']['val_split_method'] == 'k-fold':
            k = self.exp_params['train']['k']
            fl = len(self.fr_train_dataset)
            fr = list(range(fl))
            vlen = fl // k

            #get last model state if it exists
            if ls == None:
                vset_ei = fl // k
                epoch_index = 0
                val_eei = list(range(vset_ei, fl, vlen))
                si = 0
            elif ls['epoch'] == -1:
                si = ls['fold'] + vlen
                vset_ei = ls['fold'] + (2 * vlen)
                epoch_idx = 0
                val_eei = list(range(vset_ei, fl, vlen))
                if val_eei[-1] == None or val_eei[-1] + vlen < fl:
                    val_eei.append(val_eei[-1] + vlen)
            else:
                si = ls['fold']
                vset_ei = ls['fold'] + vlen
                epoch_idx = ls['epoch']
                val_eei = list(range(vset_ei, fl, vlen))
                if val_eei[-1] == None or val_eei[-1] + vlen < fl:
                    val_eei.append(val_eei[-1] + vlen)

            #get best model state if it exists
            bestm_valacc = 0.0 if bs == None else bs['valacc']
            bestm_valloss = 99999 if bs == None else bs['valloss']
            bestm_trloss = 0.0 if bs == None else 0.0
            bestm_tlh = torch.zeros(self.exp_params['train']['num_epochs']) if bs == None else bs['trlosshistory']
            bestm_vlh = torch.zeros(self.exp_params['train']['num_epochs']) if bs == None else bs['vallosshistory']
            bestm_vah = torch.zeros(self.exp_params['train']['num_epochs']) if bs == None else bs['valacchistory']
            best_model = {} if bs == None else self.__build_model().load_state_dict(bs['model_state'])
            best_fold = vset_ei
            

            for vi, ei in enumerate(val_eei):
                print(f"Running split {vi} with si: {si} and ei: {ei}")
                val_idxs = fr[si:ei]
                tr_idxs = [fi for fi in fr if fi not in val_idxs]
                train_dataset = Subset(self.fr_train_dataset, tr_idxs)
                val_dataset = Subset(self.fr_train_dataset, val_idxs)

                train_loader = DataLoader(train_dataset,
                    batch_size = self.exp_params['train']['batch_size'],
                    shuffle = False
                )
                val_loader = DataLoader(val_dataset,
                    batch_size = self.exp_params['train']['batch_size'],
                    shuffle = False
                )
                model, model_info = self.__conduct_training(model, si, train_loader, val_loader)
                si = ei
                self.all_folds_res[vi] = model_info
                if model_info["valloss"] < bestm_valacc:
                    best_model = model
                    bestm_valloss = model_info["valloss"]
                    bestm_valacc = model_info["valacc"]
                    bestm_trloss = model_info["trloss"]
                    bestm_vlh = model_info["vallosshistory"]
                    bestm_tlh = model_info["trlosshistory"]
                    bestm_vah = model_info["valacchistory"]
                    best_fold = model_info["fold"]
                model_info = {
                    'model_state': best_model.state_dict(),
                    'valloss': bestm_valloss,
                    'trloss': bestm_trloss,
                    'valacc': bestm_valacc,
                    'trlosshistory': bestm_tlh,
                    'vallosshistory': bestm_vlh,
                    'valacchistory': bestm_vah,
                    'fold': best_fold,
                    'epoch': -1,
                }
                self.save_model_checkpoint(model.state_dict(), self.optimizer.state_dict(), model_info,
                self.all_folds_res, 'best_state', True)
            del model_info['fold']
            del model_info['epoch']
            self.save_model_checkpoint(best_model.state_dict(), None, model_info, None)
            return self.all_folds_res
        elif self.exp_params['train']['val_split_method'] == 'fix-split':
            print("Running straight split")
            vp = self.exp_params['train']['val_percentage'] / 100
            vlen = int(vp * len(self.fr_train_dataset))
            val_idxs = np.random.randint(0, len(self.fr_train_dataset), vlen).tolist()
            tr_idxs = [idx not in val_idxs for idx in range(len(self.fr_train_dataset))]
            train_dataset = Subset(self.fr_train_dataset, tr_idxs)
            val_dataset = Subset(self.fr_train_dataset, val_idxs)

            train_loader = DataLoader(train_dataset,
                batch_size = self.exp_params['train']['batch_size'],
                shuffle = self.exp_params['train']['shuffle_data']
            )
            val_loader = DataLoader(val_dataset,
                batch_size = self.exp_params['train']['batch_size'],
                shuffle = self.exp_params['train']['shuffle_data']
            )
            model, model_info = self.__conduct_training(model, -1, train_loader, val_loader)
            del model_info['fold']
            del model_info['epoch']
            self.save_model_checkpoint(model.state_dict(), None, model_info, None)
            return {}
        else:
            raise SystemExit("Error: no valid split method passed! Check run.yaml")
    
    def save_model_checkpoint(self, model_state, optimizer_state, chkpt_info,
    model_history = None, chkpt_type = 'last_state'):
        if model_history == None:
            save_experiment_output(model_state, chkpt_info, self.exp_params,
                True, False)
            os.remove(os.path.join(self.root_dir), "models/checkpoints/current_model.pt")
        else:
            save_model_chkpt(model_state, optimizer_state, chkpt_info, model_history, chkpt_type)
        
    def test(self, model, test_dataset):
        model = model.cpu()
        test_loader = DataLoader(test_dataset, batch_size = self.exp_params["train"]["batch_size"], shuffle = True)
        model.eval()
        loss_fn = self.__loss_fn(self.exp_params["train"]["loss"])
        running_loss = 0.0
        acc = 0
        for _, batch in enumerate(test_loader):
            op = model(batch[self.X_key])
            loss = loss_fn(op, batch[self.y_key])
            running_loss += loss.item()
            acc = get_accuracy(op, batch[self.y_key])
        print("Loss:", running_loss/len(test_loader))
        print("Accuracy:", acc/len(test_loader), "\n")
        
     