from models.unet import UNet

def get_model(model_name = 'unet'):
    model = {}
    if model_name == "UNet":
        model = UNet()
    else:
        SystemExit("Error: no valid model name passed! Check run.yaml")
