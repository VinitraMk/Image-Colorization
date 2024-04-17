from models.unet import UNet
from models.conv_net import ConvNet

def get_model(model_name = 'unet'):
    model = {}
    if model_name == "unet":
        model = UNet()
    elif model_name == "conv_net":
        model = ConvNet()
    else:
        raise SystemExit("Error: no valid model name passed! Check run.yaml")
    return model
