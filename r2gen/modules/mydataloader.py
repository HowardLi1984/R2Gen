import os
import json
import torch
from torchvision import transforms
from PIL import Image


class mydataloader():
    def __init__(self, filepath, args):
        self.batch_size = args.batch_size
        self.filepath = filepath
        self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406),
                                     (0.229, 0.224, 0.225))])
    
    def out(self):
        # 将图片重复batch_size并返回
        image = Image.open(self.filepath).convert('RGB')
        image = self.transform(image) # 3, 224, 224
        images = [image for i in range(self.batch_size)] # [batch_size, 3, 224, 224]
        images = torch.stack(images, 0) # (batch_size, 3, 224, 224)
        return images

