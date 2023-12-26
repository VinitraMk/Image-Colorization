from torchvision.io import read_image
from PIL import Image
import torch

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
    