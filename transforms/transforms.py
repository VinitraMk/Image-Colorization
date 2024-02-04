from torchvision import transforms, utils
import torch
from common.utils import get_config

class Resize(object):
    def __init__(self, output_size = 256):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size
        
    def __call__(self, image):
        config = get_config()
        
        h, w = image.shape[:2]
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size
              
        new_h, new_w = int(new_h), int(new_w)
        image = transforms.resize(image, (new_h, new_w))
        return image
    
class RandomCrop(object):
    
    def __init__(self, output_size = 224):
        assert isinstance(output_size, (int, tuple))
        
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size
        
    def __call__(self, image):
        h, w = image.shape[:2]
        new_h, new_w = self.output_size
        
        top = torch.randint(0, h - new_h + 1)
        left = torch.randint(0, w - new_w + 1)
        image = image[top: top + new_h, left: left + new_w]
        return image

class CenterCrop(object):
    
    def __init__(self, output_size = 224):
        assert isinstance(output_size, (int, tuple))
        
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size
    
    def __call__(self, image):
        assert isinstance(image, torch.tensor)
        image = transforms.functional.center_crop(image)

        return image
        
class ToTensor(object):
    
    def __call__(self, image):
        image = torch.from_numpy(image)
        #image = image.permute((2, 0, 1))
        return image
            
            