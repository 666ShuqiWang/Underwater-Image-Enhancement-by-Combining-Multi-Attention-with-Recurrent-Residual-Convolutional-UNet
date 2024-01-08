import torch
import torch.nn as nn 
import cv2,datetime,os
from net import GeneratorNet#GeneratorNet
import argparse
import numpy as np
from utils import img2tensor,tensor2img
from tqdm import tqdm
from matplotlib import pyplot as plt
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
parser = argparse.ArgumentParser()
parser.add_argument('--img_folder',type=str,default='./PyTorch/data/1/',help='input image path')
parser.add_argument('--checkpoint',type=str,default='checkpoints/end/netG_24.pth',help='checkpoint for generator')
#parser.add_argument('--checkpoint',type=str,default='checkpoints/l1loss_only/netG_422.pth',help='checkpoint for generator')
parser.add_argument('--output_folder',type=str,default='./PyTorch/data/2/',help='output folder')
args = parser.parse_args()

if __name__ == "__main__":
    netG = GeneratorNet().to(device)
    with torch.no_grad():
        #checkpoint = torch.load(args.checkpoint)#GPU跑的 l1loss_only 501020-97lun
        checkpoint = torch.load('checkpoints/end/netG_61.pth', map_location='cpu')#改的是这里！
        #checkpoint = torch.load('checkpoints/l1loss_only/netG_422.pth', map_location='cpu')  # 改的是这里！
        netG.load_state_dict(checkpoint)
        img_folder = args.img_folder
        pbar = tqdm(os.listdir(img_folder))
        for img_name in os.listdir(img_folder):
            img_path = os.path.join(img_folder,img_name)
            img = cv2.imread(img_path)
            img = cv2.resize(img,(1024,1024))
            #img = cv2.resize(img, (512, 512))
            img_tensor = img2tensor(img)
            output_tensor = netG.forward(img_tensor)
            output_img = tensor2img(output_tensor)
            save_folder = args.output_folder
            if not os.path.exists(save_folder):
                os.makedirs(save_folder)
            save_path = os.path.join(save_folder,img_name)

            cv2.imwrite(save_path,output_img)
            pbar.update(1)
