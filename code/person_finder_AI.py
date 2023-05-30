# This program is used to train the nerual network
# ------------------------------------------------

# This is used for some of the file paths
import os
# This is used for arrays
import numpy as np
# I use this to read in images and sometimes display when testing
import cv2
# This is used for counting the time
from time import time

# These are all the pytorch imports
import torch
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

# This is tensor board which allowed me to see the loss and images during training
from torch.utils.tensorboard import SummaryWriter
# This was used to manipulate images
from PIL import Image

# These are my programs
# Utils has some functions for uploading data to tensor board
from utils import *
# This contains the data parser to read in the dataset
from data_parser import parse_data

# Data settings
# This is the size of the input images
IN_IMG_SIZE = 1024
# This is the size of the neural network ouput mask
OUT_IMG_SIZE = 256
# This is the max number of people in an image because I used arrays
MAX_NUM_OF_PEOPLE = 50

# Loss weights
# This is the weight for the cross entropy loss
W1 = 1.0
# This is the weight for the l1 loss
W2 = 20.0

# This transforms the image with some random ajustments in brightness, contrast, and saturation
# This is done so that the neural network trains on different types of images
# It also converts the image to a tensor
dataTransformations = transforms.Compose([
    transforms.ColorJitter(brightness=(0.5,1.0), contrast=(0.3,1.0), saturation=(0.3,1.0)),
    transforms.ToTensor()])

# This functions clamps a value between two numbers
def clamp(value, minValue, maxValue):
    return max(min(maxValue, value), minValue)

# This fills in a specified area in a tensor with a specific value
# Input variables
# tensor - the tensor to edit
# x and y are the top left corner of the area to change
# width and height are the width and height of the area to change
# value is what to put in that area
# This function is used to create the target mask using bounding boxes
def fill_tensor_mask(tensor, x, y, width, height, value):
    # Tensor is in HWC
    # This sets the values of the tensor within the area to the value
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
# targetsBatch is a tensor of the images in the batch
def create_mask(targetsBatch):
    # This creates the ouput tensor batch
    # This is the channel height width format with 5 channels
    masksBatch = torch.Tensor(targetsBatch.shape[0], 5, OUT_IMG_SIZE, OUT_IMG_SIZE)

    # We assume targetsBatch contains normalized bbox left,top coordinates and width and height in (0..1) range
    # Normalization is done by dividing them by IN_IMG_SIZE
    # This loops through the batch
    for j in range (0, targetsBatch.shape[0]):   # loop over batch
        # This gets the current target and scales it to the image size
        currentTarget = targetsBatch[j]*OUT_IMG_SIZE # scale to ouput size
        # This creates a mask in width height channel format
        mask = torch.zeros(OUT_IMG_SIZE, OUT_IMG_SIZE, 5)
        
        # This loops through the target
        # It loops for the max number of people
        # If there are less people than the max the data is all 0s
        for i in range (0, MAX_NUM_OF_PEOPLE):
            # Checks if the data is not equal to zero meaning it isnt a person
            if currentTarget[0 + i*4] != 0 and currentTarget[1 + i*4] != 0 and currentTarget[2 + i*4] != 0 and currentTarget[3 + i*4] != 0:
                # Sets the x, y, width, and height of the bounding box to variables that are normalized
                normBboxX = targetsBatch[j][0 + i*4]
                normBboxY = targetsBatch[j][1 + i*4]
                normBboxW = targetsBatch[j][2 + i*4]
                normBboxH = targetsBatch[j][3 + i*4]
                
                # This clamps the scaled target x, y, width, height in the out image size
                xtl = clamp(currentTarget[0 + i*4], 0, OUT_IMG_SIZE-1)
                ytl = clamp(currentTarget[1 + i*4], 0, OUT_IMG_SIZE-1)

                # These are the bottom right coordinate
                xbr = clamp(currentTarget[0 + i*4] + currentTarget[2 + i*4], 0, OUT_IMG_SIZE-1)
                ybr = clamp(currentTarget[1 + i*4] + currentTarget[3 + i*4], 0, OUT_IMG_SIZE-1)

                # This calculates the width and height using the scaled coordinates
                width = xbr - xtl
                height = ybr - ytl

                # This fills the area of the boudning box with the target in the mask tensor
                mask = fill_tensor_mask(mask, xtl, ytl, width, height, torch.Tensor([1.0, normBboxX, normBboxY, normBboxW, normBboxH]))

        # This converts the tensor to channel height width
        mask = mask.permute(2, 0, 1) 
        # This sets the correct element in the batch to the mask
        masksBatch[j] = mask

    # This returns the batch tensor with all the masks
    return masksBatch

# This draws the bounding boxes on an image using open cv
# image is the image to draw on and result is the bounding boxes
def draw_labels(image, result):    
    # This loops through result tensor if there is an empty bounding box it wont draw
    for i in range(0, MAX_NUM_OF_PEOPLE):
        # This uses the x, y, width, height from result to draw the bounding box
        image = cv2.rectangle(image, (int(result[0 + i*4]), int(result[1 + i*4])), (int(result[0 + i*4]) + int(result[2 + i*4]), int(result[1 + i*4]) + int(result[3 + i*4])), (0, 255, 0), 2)

    # And this returns the image to be displayed
    return image

# This is the dataset class which I made which pytorch uses to get data
# Py torch requires three functions the contructor, __len__ which returns the length
# of the data, and __getitem__ which needs to return an item by index
class PeopleDataset(Dataset):
    def __init__(self, pathToManifest, pathToImages, imageTransform=None):
        # First I use the functions from my data parser code to get the data
        # pathToManifest is the path to the annotations
        self.data = parse_data(pathToManifest)

        # This sets the image transform
        self.imageTransform = imageTransform
        # And this is the path to all of the iages in the dataset
        self.pathToImages = pathToImages

    # This returns the length of the data
    def __len__(self):
        # It return the length of the first axis of the array
        return self.data.shape[0]

    # This gets an element from the data by index
    def __getitem__(self, idx):
        # This gets the image path from the data
        imagePath = self.data[idx][0]
        
        # The images are stored in 3 seperate folders so the program
        # uses try and excepts to find which folder it is in and open
        # the image
        image = 0
        try:
            image = Image.open(f"data/CrowdHuman_train01/Images/{imagePath}.jpg")
        except:
            try:
                image = Image.open(f"data/CrowdHuman_train02/Images/{imagePath}.jpg")
            except:
                image = Image.open(f"data/CrowdHuman_train03/Images/{imagePath}.jpg")
        
        # It then saves the originial dimesnions of the image for scaling later on
        original_dimensions = image.size
        # This resizes the image because the nueral network has an input of 1024x1024
        image = image.resize((IN_IMG_SIZE, IN_IMG_SIZE))

        # This creates variables for the x and y scale using the original dimensions of the image
        xScale = IN_IMG_SIZE/original_dimensions[0]
        yScale = IN_IMG_SIZE/original_dimensions[1]

        # This creates an enmpy tensor that will store data for all the bounding boxes
        bbox = torch.zeros((MAX_NUM_OF_PEOPLE*4))
        # This loops for the max number of people
        for i in range (0, MAX_NUM_OF_PEOPLE):
            # This sets the four values in the tensor to x, y, width, and height
            bbox[i*4] = self.data[idx][1][i*4]*xScale
            bbox[i*4 + 1] = self.data[idx][1][i*4 + 1]*yScale
            bbox[i*4 + 2] = self.data[idx][1][i*4 + 2]*xScale
            bbox[i*4 + 3] = self.data[idx][1][i*4 + 3]*yScale
            #image = cv2.rectangle(image, (int(bbox[0]*xScale), int(bbox[1]*yScale)), (int(xScale*(bbox[0] + bbox[2])), int(yScale*(bbox[1] + bbox[3]))), (255, 0, 0), 4)
        
        # This uses the image transform to convert the image to a tensor
        imageAsTensor = self.imageTransform(image)

        # Normalizes the bounding boxes by dividing by the in image size
        targets = bbox / IN_IMG_SIZE
        
        # This return the image which is the input to the neural network
        # and the bounding boxoes which are the target
        return imageAsTensor, targets

# This is the class for the actual neural network
class PersonNet(nn.Module):
    def __init__(self):
        super().__init__()

        # This creates all of the layers of the neural network
        # First uses convolutional layers shrinking the image down to 64x64
        # Then it uses deconvolutional layers to enlarge the image to the out image size

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

        # This creates a list of all the layers which can be used to print information
        # about the neural network
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
        # This functions is what actually allows the neural network to input something
        # through all of the layers

        # It just has a variable x which it inputs through the layers of the neural network
        # I also use the elu function on x
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
        # These are some of the skip connections

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
        
        # This checks if the ouput is in a batch or just a lone image
        # this is done to determine how to run sigmoid on the values
        # Sigmoid is used to keep the values from 0 to 1
        if x.dim() == 4:
            x[:, 0:1, :, :] = F.sigmoid(x[:, 0:1, :, :])    # out 256x256x1
        else:
            x[0:1, :, :] = F.sigmoid(x[0:1, :, :])

        # It then returns x
        return x

# This loads the neural network using a path
def load_model(path):
    model = PersonNet()
    model.load_state_dict(torch.load(path))
    model.eval()

    return model

# This draws the bounding boxes while filtering them
def draw_result_bboxes(image, result):    
    # First it gets a threshhold, only values of more than 0.8 make it through
    # this filters any random pixels out
    filteredResult = nn.Threshold(0.8, 0.0)(result[0])
    # It then creates a tensor with the indexes of all the non zero values
    nonZeroIndices = torch.nonzero(filteredResult)

    # Then it loops through all of the indecies
    for i in range (0, nonZeroIndices.shape[0]):        
        # This gets the x, y, width, height at the index
        resultVector = result[:, nonZeroIndices[i, 0], nonZeroIndices[i, 1]]

        # This sets the x, y, width, and height and scales it to the correct size
        bboxX = int(resultVector[1]*IN_IMG_SIZE) + int((nonZeroIndices[i, 1]/OUT_IMG_SIZE)*IN_IMG_SIZE)
        bboxY = int(resultVector[2]*IN_IMG_SIZE) + int((nonZeroIndices[i, 0]/OUT_IMG_SIZE)*IN_IMG_SIZE)
        bboxW = int(resultVector[3]*IN_IMG_SIZE)
        bboxH = int(resultVector[4]*IN_IMG_SIZE)

        # This draws the bounding box on the image
        image = cv2.rectangle(image, (bboxX, bboxY), (bboxX + bboxW, bboxY + bboxH), color=(255, 0, 0), thickness=1)

    return image
            
# This function is the main function that actually trains the nerual networks
# device is the device to run it on usually the graphics card
# start epoch is the epoch to start training on
# n_epochs is the number of epochs to train for
# pathToManifest is the path to the annotations
# pathToData is where the images are stored
# pathToLogs is where to save logs os the training for tensor board and models
# pathToModel is the path of the model to load
def train(device, start_epoch, n_epochs, pathToManifest, pathToData, pathToLogs, pathToModel = None):
    # This sets te weights for the losses which will be used later
    w1 = W1
    w2 = W2
    
    # This is an object which is used to upload things to tensor board
    writer = SummaryWriter(log_dir=pathToLogs)

    # This creates an object for the dataset
    # The batch size is 16 and it shuffles the dataset
    dataLoader = DataLoader( PeopleDataset(imageTransform=dataTransformations, 
        pathToManifest=pathToManifest, 
        pathToImages=pathToData), 
        batch_size=16, shuffle=True)

    # This creates the model by loading the neural network and converting
    # it to the device, so that it runs faster
    # I will assume that the device is the graphics card but it works on cpu as well
    # just slower
    model = PersonNet().to(device)
    # If there is no path specified then it creates a new model
    if pathToModel != None :
        model = load_model(pathToModel).to(device)

    # This prints out information about the model including the number of parameters
    print("Model:")
    print(model)
    modelTotalParams = sum(p.numel() for p in model.parameters())
    print("Number of parameters in the model: ", modelTotalParams)

    # This creates an object for the optimizer
    # Usually I would use SGD which is stacastic gradient ddesent
    # but AdamW seems to work a bit better with this dataset
    # I trained most of the dataset with a learning rate of 1e-4 but
    # towards the end I set it to 1e-6 to fine tune the weights
    optimizer = optim.AdamW(model.parameters(), lr=1e-6)

    # This gets the time that the epoch started training
    startTime = time()
    # This is a batch counter
    idx_counter = len(dataLoader.dataset)*(start_epoch - 1) # epochs start with 1
    
    # This is a loop that loops for the amount of epochs specified
    for epoch_index in range (start_epoch, start_epoch + n_epochs):
        # This just prints out the epoch index and the learning rate
        print('Epoch {}, lr {}'.format(epoch_index, optimizer.param_groups[0]['lr']))

        # This runs every 5 epochs
        if epoch_index % 5 == 0:
            # It logs weight and gradient histograms to tensor board
            log_weight_histograms(writer, epoch_index, model)
            log_gradient_histograms(writer, epoch_index, model)

        # This loops through batches in the dataset
        for images, targets in dataLoader:
            # First it sets the batch images and batch targets to variables
            # and it converts them onto the GPU
            batchImages = images.to(device)
            batchTargets = targets.to(device)

            # This converts the targets to masks using the create_mask function
            # and converts it to the GPU
            mask_tensor = create_mask(batchTargets).to(device)

            # This clears the gradients
            optimizer.zero_grad()

            # Then it inputs the image into the neural network and gets an output
            batchOuput = model(batchImages)
            # It thens runs binary cross entropy loss on the mask portion which is the first channel of the output image
            loss_ce = nn.BCELoss(reduction="mean")(batchOuput[:,0:1,:,:], mask_tensor[:,0:1,:,:])
            # And then it runs l1 loss on the bounding boxes
            loss_mse = nn.SmoothL1Loss(reduction="mean")(batchOuput[:,1:,:,:], mask_tensor[:,1:,:,:])
            # It then adds the weights and multiplies the weights for training
            total_loss = w1*loss_ce + w2*loss_mse

            # Then it uses the total loss to adjust all the values in the nerual network
            total_loss.backward()
            optimizer.step()

            # Every 50 batches it logs data
            if idx_counter % 50 == 0:
                # Logging
                # First it los the original image that was inputed into the neural network
                # All these things just convert the image from a tensor into a displayable image
                image_to_log = images[0].permute(1, 2, 0).cpu().detach().numpy()
                image_to_log *= 255.0   # de-normalize
                image_to_log = image_to_log.astype(np.uint8).copy()

                # This draws the image with the bounding boxes   
                labels = targets[0].cpu().detach().numpy()
                labels *= IN_IMG_SIZE # de-normalize
                image_with_labels = draw_labels(image_to_log.copy(), labels)
                image_with_results = draw_result_bboxes(image_to_log.copy(), batchOuput[0].cpu())

                # This uploads the cross entropy loss
                writer.add_scalar("Loss_ce", loss_ce, idx_counter)
                # This uploads the l1 loss
                writer.add_scalar("Loss_mse", loss_mse, idx_counter)
                # This uploads the total loss
                writer.add_scalar("Total_Loss", total_loss, idx_counter)
                # This uploads the learning rate
                writer.add_scalar("LR", optimizer.param_groups[0]['lr'], idx_counter)
                
                # This uploads the original image with correct bounding boxes
                writer.add_image("image_labels", transforms.ToTensor()(image_with_labels), idx_counter)
                # This uploads the ouput image with the ouput bounding boxes
                writer.add_image("result_bboxes", transforms.ToTensor()(image_with_results), idx_counter)
                # This uploads the target mask created
                writer.add_image("target_mask", mask_tensor[0][0], idx_counter, dataformats="HW")
                # This uploads the mask the the neural network created 
                writer.add_image("result_mask", batchOuput[0][0], idx_counter, dataformats="HW")
                # This uploads all the data to tensor board
                writer.flush()
            
            # It increments the batch counter
            idx_counter += 1
            # This prints out the epoch and the total loss that epoch
            print("Epoch: ", epoch_index, " Total Loss: ", total_loss.item())

        # Save model every epoch
        if epoch_index % 1 == 0:
            torch.save(model.state_dict(), os.path.join(pathToLogs, "facenet_{}.pt".format(epoch_index))) 


    # Closes the writer
    writer.close()
    # Prints out the total training time
    print("Training Time: ", (time() - startTime)/60)

# This is the main function that runs the train function
def main():
    # First it creates the device
    # If a graphics card is available it uses that otherwise it uses the CPU
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # This runs the train function with all of the prameters
    train(device, start_epoch = 70, n_epochs = 40,
          pathToManifest="data/annotation_train.odgt",
          pathToData="Data/WIDER_train/images/",
          pathToLogs="D:\Person_Finder_Logs",
          pathToModel = "D:/Person_Finder_Logs/facenet_69.pt")

# This runs main
if __name__ == "__main__":
    main()