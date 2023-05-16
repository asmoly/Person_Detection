import os
import numpy as np
import pickle
import random
import cv2
import json
from time import time

import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

from torch.utils.tensorboard import SummaryWriter
from PIL import Image

from utils import *
from data_parser import parse_data

# Data settings
IN_IMG_SIZE = 1024
OUT_IMG_SIZE = 256
MAX_NUM_OF_FACES = 50

# Loss weights
W1 = 1.0
W2 = 20.0 # make 1.0 in the begining and then 100.0 for final bbox training!

dataTransformations = transforms.Compose([
    transforms.ColorJitter(brightness=(0.5,1.0), contrast=(0.3,1.0), saturation=(0.3,1.0)),
    transforms.ToTensor()])

def convert_to_tensor(image):
    return dataTransformations(image)

def clamp(value, minValue, maxValue):
    return max(min(maxValue, value), minValue)

def fill_tensor_mask(tensor, x, y, width, height, value):
    # Tensor is in HWC
    tensor[int(y):int(y) + int(height), int(x):int(x) + int(width)] = value

    indexTensor = torch.Tensor(np.indices((OUT_IMG_SIZE, OUT_IMG_SIZE))).permute(1, 2, 0)/OUT_IMG_SIZE
    bboxMask = torch.zeros((OUT_IMG_SIZE, OUT_IMG_SIZE, 2))
    bboxMask[int(y):int(y) + int(height), int(x):int(x) + int(width)] = torch.Tensor([1.0, 1.0])
    indexTensor *= bboxMask

    # index tensor has (y, x) and tensor has (x, y)
    tensor[:, :, 1] -= indexTensor[:, :, 1]
    tensor[:, :, 2] -= indexTensor[:, :, 0]

    return tensor

# Creates target mask at output resolution with 5 channels: face_score, x, y, width, height 
def create_mask(targetsBatch):
    masksBatch = torch.Tensor(targetsBatch.shape[0], 5, OUT_IMG_SIZE, OUT_IMG_SIZE)

    # We assume targetsBatch contains normalized bbox left,top coordinates and width and height in (0..1) range
    # Normalization is done by dividing them by IN_IMG_SIZE
    for j in range (0, targetsBatch.shape[0]):   # loop over batch
        currentTarget = targetsBatch[j]*OUT_IMG_SIZE # scale to ouput size
        mask = torch.zeros(OUT_IMG_SIZE, OUT_IMG_SIZE, 5)
        
        for i in range (0, MAX_NUM_OF_FACES):
            #print(currentTarget)
            if currentTarget[0 + i*4] != 0 and currentTarget[1 + i*4] != 0 and currentTarget[2 + i*4] != 0 and currentTarget[3 + i*4] != 0:
                normBboxX = targetsBatch[j][0 + i*4]
                normBboxY = targetsBatch[j][1 + i*4]
                normBboxW = targetsBatch[j][2 + i*4]
                normBboxH = targetsBatch[j][3 + i*4]
                
                xtl = clamp(currentTarget[0 + i*4], 0, OUT_IMG_SIZE-1)
                ytl = clamp(currentTarget[1 + i*4], 0, OUT_IMG_SIZE-1)

                xbr = clamp(currentTarget[0 + i*4] + currentTarget[2 + i*4], 0, OUT_IMG_SIZE-1)
                ybr = clamp(currentTarget[1 + i*4] + currentTarget[3 + i*4], 0, OUT_IMG_SIZE-1)

                width = xbr - xtl
                height = ybr - ytl

                mask = fill_tensor_mask(mask, xtl, ytl, width, height, torch.Tensor([1.0, normBboxX, normBboxY, normBboxW, normBboxH]))

        mask = mask.permute(2, 0, 1)    # convert to CHW
        masksBatch[j] = mask

    return masksBatch

def draw_labels(image, result):    
    for i in range(0, MAX_NUM_OF_FACES):
        image = cv2.rectangle(image, (int(result[0 + i*4]), int(result[1 + i*4])), (int(result[0 + i*4]) + int(result[2 + i*4]), int(result[1 + i*4]) + int(result[3 + i*4])), (0, 255, 0), 2)

    return image


class PeopleDataset(Dataset):
    def __init__(self, pathToManifest, pathToImages, imageTransform=None):
        self.data = parse_data(pathToManifest)

        self.imageTransform = imageTransform
        self.pathToImages = pathToImages

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        imagePath = self.data[idx][0]
        
        image = 0

        try:
            image = Image.open(f"data/CrowdHuman_train01/Images/{imagePath}.jpg")
        except:
            try:
                image = Image.open(f"data/CrowdHuman_train02/Images/{imagePath}.jpg")
            except:
                image = Image.open(f"data/CrowdHuman_train03/Images/{imagePath}.jpg")
        
        original_dimensions = image.size
        #print(original_dimensions)
        image = image.resize((1024, 1024))
        #image = np.array(image)

        xScale = 1024/original_dimensions[0]
        yScale = 1024/original_dimensions[1]

        bbox = torch.zeros((50*4))
        for i in range (0, 50):
            bbox[i*4] = self.data[idx][1][i*4]*xScale
            bbox[i*4 + 1] = self.data[idx][1][i*4 + 1]*yScale
            bbox[i*4 + 2] = self.data[idx][1][i*4 + 2]*xScale
            bbox[i*4 + 3] = self.data[idx][1][i*4 + 3]*yScale
            #image = cv2.rectangle(image, (int(bbox[0]*xScale), int(bbox[1]*yScale)), (int(xScale*(bbox[0] + bbox[2])), int(yScale*(bbox[1] + bbox[3]))), (255, 0, 0), 4)
        

        imageAsTensor = self.imageTransform(image)

        # Normalize targets (x,y,width,height) to 0..1 range in absolute coords
        targets = bbox / IN_IMG_SIZE
        
        return imageAsTensor, targets

class PersonNet(nn.Module):
    def __init__(self):
        super().__init__()

        # in IN_IMG_SIZExIN_IMG_SIZE: 1024x1024
        self.enc_in_conv1_ds = nn.Conv2d(3, 16, kernel_size=5, stride=2, padding=2)     # out 512x512
        # block1
        self.enc_b1_conv1 = nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1)       # out 512x512
        self.enc_b1_conv2 = nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1)       # out 512x512
        # block2
        self.enc_b2_conv1_ds = nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1)    # out 256x256
        self.enc_b2_conv2 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)       # out 256x256
        self.enc_b2_conv3 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)       # out 256x256
        # block3
        self.enc_b3_conv1_ds = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)    # out 128x128
        self.enc_b3_conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)       # out 128x128
        self.enc_b3_conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)       # out 128x128
        # block4
        self.enc_b4_conv1_ds = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)   # out 64x64
        self.enc_b4_conv2 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)     # out 64x64
        self.enc_b4_conv3 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)     # out 64x64

        self.enc_latent_conv1 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)   # out 64x64

        self.dec_deconv1 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1)    # out 128x128
        # add skip - enc_b3_conv3
        self.dec_conv1 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)  # out 128x128
        self.dec_conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)  # out 128x128
        self.dec_deconv2 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1) # out 256x256
        # add skip - enc_b2_conv3
        self.dec_conv3 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)  # out 256x256
        self.dec_conv4 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)  # out 256x256
        
        self.dec_out_conv1 = nn.Conv2d(32, 16, kernel_size=3, stride=1, padding=1)   # out 256x256
        self.dec_out_conv2 = nn.Conv2d(16, 5, kernel_size=3, stride=1, padding=1)    # out 256x256

        self.layers = [self.enc_in_conv1_ds, 
                       self.enc_b1_conv1,
                       self.enc_b1_conv2,
                       self.enc_b2_conv1_ds,
                       self.enc_b2_conv2,
                       self.enc_b2_conv3,
                       self.enc_b3_conv1_ds,
                       self.enc_b3_conv2,
                       self.enc_b3_conv3,
                       self.enc_b4_conv1_ds,
                       self.enc_b4_conv2,
                       self.enc_b4_conv3,
                       self.enc_latent_conv1,
                       self.dec_deconv1,
                       self.dec_conv1,
                       self.dec_conv2,
                       self.dec_deconv2,
                       self.dec_conv3,
                       self.dec_conv4,
                       self.dec_out_conv1,
                       self.dec_out_conv2]

    def forward(self, x):    
        x = F.elu(self.enc_in_conv1_ds(x))

        x = F.elu(self.enc_b1_conv1(x))
        x = F.elu(self.enc_b1_conv2(x))

        x = F.elu(self.enc_b2_conv1_ds(x))
        x = F.elu(self.enc_b2_conv2(x))
        x = F.elu(self.enc_b2_conv3(x)) 
        skip_x_1 = x    # out 256x256x32

        x = F.elu(self.enc_b3_conv1_ds(x))
        x = F.elu(self.enc_b3_conv2(x))
        x = F.elu(self.enc_b3_conv3(x))
        skip_x_2 = x    # out 128x128x64

        x = F.elu(self.enc_b4_conv1_ds(x))
        x = F.elu(self.enc_b4_conv2(x))
        x = F.elu(self.enc_b4_conv3(x))

        x = F.elu(self.enc_latent_conv1(x))

        x = F.elu(self.dec_deconv1(x))  # out 128x128x64
        x = x + skip_x_2
        x = F.elu(self.dec_conv1(x))
        x = F.elu(self.dec_conv2(x))
        x = F.elu(self.dec_deconv2(x))  # out 256x256x32
        x = x + skip_x_1
        x = F.elu(self.dec_conv3(x))
        x = F.elu(self.dec_conv4(x))

        x = F.elu(self.dec_out_conv1(x))
        x = self.dec_out_conv2(x)
        
        if x.dim() == 4:
            x[:, 0:1, :, :] = F.sigmoid(x[:, 0:1, :, :])    # out 256x256x1
        else:
            x[0:1, :, :] = F.sigmoid(x[0:1, :, :])

        return x        #return x = self.layers(x) # if using sequential

def load_model(path):
    model = PersonNet()
    model.load_state_dict(torch.load(path))
    model.eval()

    return model

def draw_result_bboxes(image, result):    
    filteredResult = nn.Threshold(0.8, 0.0)(result[0])
    nonZeroIndices = torch.nonzero(filteredResult)      # returns a list of (y,x) positions where result passed the threshold

    for i in range (0, nonZeroIndices.shape[0]):        
        resultVector = result[:, nonZeroIndices[i, 0], nonZeroIndices[i, 1]]

        # Result is: vector (x,y) pointing from pixel to top/left bbox corner and (W,H) of bbox. All 0..1 normalized
        # Need to properly de-normalize and convert to absolute frame coords
        bboxX = int(resultVector[1]*IN_IMG_SIZE) + int((nonZeroIndices[i, 1]/OUT_IMG_SIZE)*IN_IMG_SIZE)
        bboxY = int(resultVector[2]*IN_IMG_SIZE) + int((nonZeroIndices[i, 0]/OUT_IMG_SIZE)*IN_IMG_SIZE)
        bboxW = int(resultVector[3]*IN_IMG_SIZE)
        bboxH = int(resultVector[4]*IN_IMG_SIZE)

        image = cv2.rectangle(image, (bboxX, bboxY), (bboxX + bboxW, bboxY + bboxH), color=(255, 0, 0), thickness=1)

    return image
            
# start_epoch = 1 at the very begining!
def train(device, start_epoch, n_epochs, pathToManifest, pathToData, pathToLogs, pathToModel = None):
    w1 = W1
    w2 = W2
    
    writer = SummaryWriter(log_dir=pathToLogs)

    dataLoader = DataLoader( PeopleDataset(imageTransform=dataTransformations, 
        pathToManifest=pathToManifest, 
        pathToImages=pathToData), 
        batch_size=16, shuffle=True)

    # validationSet = DataLoader(PeopleDataset(imageTransform=dataTransformations, 
    #                                         pathToManifest="C:\Sasha\Programing_Robotics\AI\Face_Detection\Data\wider_face_split/wider_face_val_bbx_gt.txt",
    #                                         pathToImages="C:\Sasha\Programing_Robotics\AI\Face_Detection\Data\WIDER_val\images"),
    #                                         batch_size=16, shuffle=True)
    
    model = PersonNet().to(device)
    if pathToModel != None :
        model = load_model(pathToModel).to(device)

    print("Model:")
    print(model)
    modelTotalParams = sum(p.numel() for p in model.parameters())
    print("Number of parameters in the model: ", modelTotalParams)

    optimizer = optim.AdamW(model.parameters(), lr=1e-6)        #optim.SGD(model.parameters(), lr=0.0001)
    #scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[20,40,60], gamma=0.1)

    startTime = time()
    idx_counter = len(dataLoader.dataset)*(start_epoch - 1) # epochs start with 1
    for epoch_index in range (start_epoch, start_epoch + n_epochs):
        print('Epoch {}, lr {}'.format(epoch_index, optimizer.param_groups[0]['lr']))

        #if epoch_index % 5 == 0:
            # Log weight and gradient histograms
            #log_weight_histograms(writer, epoch_index, model)
            #log_gradient_histograms(writer, epoch_index, model)

        for images, targets in dataLoader:
            batchImages = images.to(device)
            batchTargets = targets.to(device)

            mask_tensor = create_mask(batchTargets).to(device)

            optimizer.zero_grad()

            batchOuput = model(batchImages)
            loss_ce = nn.BCELoss(reduction="mean")(batchOuput[:,0:1,:,:], mask_tensor[:,0:1,:,:])
            loss_mse = nn.SmoothL1Loss(reduction="mean")(batchOuput[:,1:,:,:], mask_tensor[:,1:,:,:])
            total_loss = w1*loss_ce + w2*loss_mse

            total_loss.backward()
            optimizer.step()

            if idx_counter % 50 == 0:
                # Logging
                image_to_log = images[0].permute(1, 2, 0).cpu().detach().numpy()
                image_to_log *= 255.0   # de-normalize
                image_to_log = image_to_log.astype(np.uint8).copy()

                labels = targets[0].cpu().detach().numpy()
                labels *= IN_IMG_SIZE # de-normalize
                
                image_with_labels = draw_labels(image_to_log.copy(), labels)
                image_with_results = draw_result_bboxes(image_to_log.copy(), batchOuput[0].cpu())

                writer.add_scalar("Loss_ce", loss_ce, idx_counter)
                writer.add_scalar("Loss_mse", loss_mse, idx_counter)
                writer.add_scalar("Total_Loss", total_loss, idx_counter)
                writer.add_scalar("LR", optimizer.param_groups[0]['lr'], idx_counter)
                
                writer.add_image("image_labels", transforms.ToTensor()(image_with_labels), idx_counter)
                writer.add_image("result_bboxes", transforms.ToTensor()(image_with_results), idx_counter)
                writer.add_image("target_mask", mask_tensor[0][0], idx_counter, dataformats="HW")
                writer.add_image("result_mask", batchOuput[0][0], idx_counter, dataformats="HW")
                writer.flush()
            
            idx_counter += 1
            print("Epoch: ", epoch_index, " Total Loss: ", total_loss.item())

        # images, targets = next(iter(validationSet))
        # images, targets = images.to(device), targets.to(device)

        # batchMasks = create_mask(targets).to(device)

        # with torch.no_grad():
        #     batchOuputVal = model(batchImages)
        #     loss_ce_val = nn.BCELoss(reduction="mean")(batchOuputVal[:,0:1,:,:], mask_tensor[:,0:1,:,:])
        #     loss_mse_val = nn.SmoothL1Loss(reduction="mean")(batchOuputVal[:,1:,:,:], mask_tensor[:,1:,:,:])
        #     total_loss_val = w1*loss_ce_val + w2*loss_mse_val

        #     writer.add_scalar("Validation_Loss", total_loss_val, idx_counter)
        #     print("Validation Set Loss: ", total_loss_val)

        # Save model checkpoint every 5th epoch
        if epoch_index % 1 == 0:
            torch.save(model.state_dict(), os.path.join(pathToLogs, "facenet_{}.pt".format(epoch_index))) 

        #scheduler.step()    # move every epoch

    writer.close()
    print("Training Time: ", (time() - startTime)/60)


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    train(device, start_epoch = 70, n_epochs = 40,
          pathToManifest="data/annotation_train.odgt",
          pathToData="Data/WIDER_train/images/",
          pathToLogs="D:\Person_Finder_Logs",
          pathToModel = "D:/Person_Finder_Logs/facenet_69.pt")

if __name__ == "__main__":
    main()