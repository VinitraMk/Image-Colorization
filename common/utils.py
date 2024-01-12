from torchvision.io import read_image
from PIL import Image
import torch
import os
import yaml
import json
from datetime import datetime

root_dir = ''
config_params = {}

def save_df_to_csv(df, filename, columns = []):
    if len(columns) == 0:
        columns = df.columns.tolist()
    df.to_csv(filename, index = False)
    
def read_imgtensor(imgpath):
    return read_image(imgpath)

def read_imgnp(imgpath):
    img = Image.open(imgpath)
    return img

def img2tensor(imgnparr):
    return torch.from_numpy(imgnparr)
   
def join_path(path_a, path_b):
    return os.path.join(path_a, path_b)

def read_yaml(ypath):
    yml_params = {}
    with open(ypath, "r") as stream:
        try:
            yml_params = yaml.safe_load(stream)
        except yaml.YAMLError as err:
            print(err)
    return yml_params

def dump_yaml(ypath, datadict):
    with open(ypath, 'w') as outfile:
        yaml.dump(datadict, outfile, default_flow_style=False)

def init_config():
    root_dir = os.getcwd()
    data_dir = os.path.join(root_dir, 'data')
    op_dir = os.path.join(root_dir, 'output')
    config_path = os.path.join(root_dir, 'config.yaml')
    config_params = read_yaml(config_path)
    config_params['root_dir'] = root_dir
    config_params['data_dir'] = data_dir
    config_params['output_dir'] = op_dir
    config_params['use_gpu'] = torch.cuda.is_available()
    dump_yaml(config_path, config_params)   

def get_exp_params():
    yaml_fp = os.path.join(root_dir, 'run.yaml')
    exp_params = read_yaml(yaml_fp)
    return exp_params

def get_config():
    config_path = os.path.join(root_dir, 'config.yaml')
    config_params = read_yaml(config_path)
    return config_params

def save2config(key, val):
    config_params = get_config()
    config_params[key] = val
    config_path = os.path.join(root_dir, 'config.yaml')
    dump_yaml(config_path, config_params)

def get_accuracy(y_pred, y_true):
    c = torch.sum(y_pred == y_true)
    return c / len(y_true)

def get_error(y_pred, y_true):
    c = torch.sum(y_pred != y_true)
    return c / len(y_true)

def save_model(model_state, chkpt_info, chkpt_filename = 'last_model', is_checkpoint = True, best_model = False):
    config_params = get_config()
    if is_checkpoint:
        fpath = os.path.join(config_params['root_dir'], 'models/checkpoints')
    else:
        if best_model:
            fpath = os.path.join(config_params['output_dir'], 'experiment_results/best_models')
        else:
            fpath = os.path.join(config_params['output_dir'], 'experiment_results/checkpoints')
    
    mpath = os.path.join(fpath, f'{chkpt_filename}.pt')
    jpath = os.path.join(fpath, f'{chkpt_filename}.json')
    torch.save(
        model_state,
        mpath
    )
    with open(jpath, 'w') as fp:
        json.dump(chkpt_info, fp)
        
def load_modelpt(model_path):
    return torch.load(model_path)

def get_modelinfo(json_filename, is_chkpt = True, is_best = False):
    model_info = {}
    cfg = get_config()
    if is_chkpt:
        json_path = os.path.join(cfg["root_dir"], "models/checkpoints/last_model.json")
    else:
        if is_best:
            json_path = os.path.join(cfg["output"], f"experiment_results/best_experiments/{json_filename}.json")
        else:
            json_path = os.path.join(cfg["output"], f"experiment_results/experiments/{json_filename}.json")
    with open(json_path, 'r') as fp:
        model_info = json.load(fp)
    return model_info

def get_model_filename(model_name):
    now = datetime.now()
    nowstr = now.strftime("%d%m%Y%H%M%S")
    fname = f'{model_name}_{nowstr}'
    return fname

def save_experiment_output(model_state, chkpt_info, exp_params, is_chkpoint = True, save_as_best = False):
    model_info = {
        'experiment_params': exp_params,
        'results': {
            'valloss': chkpt_info['valloss'],
            'trloss': chkpt_info['trloss'],
            'valacc': chkpt_info['valacc'].item(),
            'trlosshistory': chkpt_info['trlosshistory'].tolist(),
            'vallosshistory': chkpt_info['vallosshistory'].tolist(),
            'valacchistory': chkpt_info['valacchistory'].tolist(),
            'epoch': -1,
            'fold': -1
        }
    }
    save_model(model_state, model_info,
        f'last_model', is_chkpoint, save_as_best)

def save_experiment_chkpt(model_state, optimizer_state, chkpt_info, model_history, chkpt_type = "last_state"):
    cfg = get_config()
    mpath = os.path.join(cfg["root_dir"], "models/checkpoints/current_model.pt")
    if os.path.exists(mpath):
        saved_model = load_modelpt(mpath)
    else:
        saved_model = {
            "model_complete": False,
            "model_history": model_history,
            "best_state": None,
            "last_state": None
        }
    saved_model[chkpt_type] = chkpt_info
    saved_model["model_state"] = model_state #always the last state
    saved_model["optimizer_state"] = optimizer_state #always the last state
    torch.save(saved_model, mpath)

def get_saved_model(model, model_filename = '', is_chkpt = True, is_best = False):
    cfg = get_config()
    if is_chkpt:
        model_dict = load_modelpt(os.path.join(cfg["root_dir"], "models/checkpoints/last_model.pt"))
    else:
        if is_best:
            model_dict = load_modelpt(os.path.join(cfg["output_dir"], f"experiment_results/best_experiments/{model_filename}.pt"))
        else:
            model_dict = load_modelpt(os.path.join(cfg["output_dir"], f"experiment_results/experiments/{model_filename}.pt"))
    model_state = model.state_dict()
    for key in model_dict:
        model_state[key] = model_dict[key]
    return model

