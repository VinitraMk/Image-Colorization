from models.unet import UNet

def get_model(model_name = 'unet'):
    model = {}
    if model_name == "unet":
        model = UNet()
    else:
        raise SystemExit("Error: no valid model name passed! Check run.yaml")
    return model
