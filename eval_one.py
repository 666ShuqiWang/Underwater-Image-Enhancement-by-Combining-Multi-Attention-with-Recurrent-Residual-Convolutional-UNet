import torch
import torch.nn as nn 
import cv2,datetime,os
from net import GeneratorNet
import argparse
import numpy as np
from utils import img2tensor,tensor2img
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
parser = argparse.ArgumentParser()
parser.add_argument('--img_path',type=str,default='n01496331_49.jpg',help='Input the image path')
parser.add_argument('--checkpoint',type=str,default='checkpoints/netG_299.pth',help='checkpoint for generator')
args = parser.parse_args()

if __name__ == "__main__":
    netG = GeneratorNet().to(device)
    with torch.no_grad():
        checkpoint = torch.load(args.checkpoint)
        netG.load_state_dict(checkpoint)
        img_path = args.img_path
        #img ="n01496331_49.jpg"
        img = cv2.imread(img_path)
        img_tensor = img2tensor(img)
        output_tensor = netG.forward(img_tensor)
        output_img = tensor2img(output_tensor)
        cv2.imwrite('output.jpg',output_img)

