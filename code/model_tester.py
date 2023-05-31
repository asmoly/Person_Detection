import cv2

import torchvision.transforms.functional as fn
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as f

import numpy as np

from PIL import Image
import person_finder_AI

# Make sure that the camera is correct
CAMERA = 0

IN_IMG_SIZE = 1024
OUT_IMG_SIZE = 256

# This functions can calculate the percentage that one rectangle is
# overlaping with the other, the reason I did percentage instead of
# area is because percentage will work from any distance
# it takes in ax, ay, ax1, ay1 for the first rectangle
# these are the top left coord and bottom left coord
# and all the b's are for the second rectangle
def rectangle_overlap(ax, ay, ax1, ay1, bx, by, bx1, by1):
    # It then calculates the overlaping area in pixels using min max
    # by finding the two points closest to each other
    dx = min(ax1, bx1) - max(ax, bx)
    dy = min(ay1, by1) - max(ay, by)

    # This calculates the area of rectangle a
    rectangle1_area = (ax1 - ax)*(ay1 - ay)

    # It then checks dx and dy are greater than 0 then it multiplies them to get the area
    # otherwise it sets it equal to 0
    overlaping_area = 0
    if dx >= 0 and dy >= 0:
        overlaping_area =  dx*dy
    
    # It then return the overlaping area divided by the area of rectangle a
    # to get the percentage
    return overlaping_area/rectangle1_area

print("Starting person counter")

print("Opening camera 0")
# First it opens the camera using open cv
cam = cv2.VideoCapture(CAMERA, cv2.CAP_DSHOW)
# And it sets the resolution
cam.set(cv2.CAP_PROP_FRAME_WIDTH, IN_IMG_SIZE)
cam.set(cv2.CAP_PROP_FRAME_HEIGHT, IN_IMG_SIZE)
print("Opened camera")

print("Opening device")
# This opens the GPU if it is available otherwise it opens the CPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Loaded device", device)

print("Loading model")
# This loads in the model from the specified path
model = person_finder_AI.load_model("personnet_72.pt").to(device)
print("Loaded model")

# This is the main loop
while True:
    # First it reads in the camera frame
    ret, image = cam.read()

    # It then converts the image to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # image is RGB now
    # And it converts it to an array and crops it to the correct image size
    image = Image.fromarray(image)
    image = image.crop((0, 0, IN_IMG_SIZE, IN_IMG_SIZE))
    # And finally it transforms it into a tensor
    image = transforms.ToTensor()(image)

    # This means it wont calculate the gradient making it faster
    with torch.no_grad():
        # This inputs the image into the model, gets the ouput,
        # and converts it to the CPU
        modelOutput = model(image.to(device)).cpu()
        # This threshholds the output so that only person_values of 0.98 or higher stay
        modelOutput[0] = nn.Threshold(0.98, 0.0)(modelOutput[0])

        # Seperating mask from model output and resising it
        outputMask = modelOutput[0:1]
        outputMask = fn.resize(transforms.ToPILImage()(outputMask), size=[1024])
        outputMask = transforms.ToTensor()(outputMask)

        # This is for if you want to see the mask
        # cv2.imshow("test", (outputMask*255).permute(1, 2, 0).numpy().astype(np.uint8))
        # cv2.waitKey(1)

        # Drawing mask on original image
        # Creates empty tensor where to save the output image
        outputImage = torch.Tensor(3, 1024, 1024)
        # It then adds the original image and makes from 0 to 255
        outputImage[0:3] = (image*255.0)[0:3]   
        # Then it adds the ouput mask to the green chanel so that the mask apears
        # green on the image
        outputImage[1:2] += outputMask*100.0           
        
        # Formating output image
        # Clamps the color values between 0 and 255
        outputImage = torch.clamp(outputImage, 0.0, 255.0)
        # Converts it to height width channel
        outputImage = outputImage.permute(1, 2, 0)  # outputImage is HWC with RGBA
        # converts to array and convert color to BGR
        outputImage = np.asarray(outputImage).astype(np.uint8).copy() 
        outputImage = cv2.cvtColor(outputImage, cv2.COLOR_RGB2BGR)

        # This converts the ouput mask to numpy array
        resultMask = (modelOutput[0]*255).cpu().detach().numpy().astype(np.uint8)
        # It then creates an empty array with the input resolution
        mask = np.zeros(image.shape, dtype=np.uint8)
        thresh = cv2.threshold(resultMask, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

        # This find contours and finds blobs of pixels in the image
        blobs = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        blobs = blobs[0] if len(blobs) == 2 else blobs[1]

        bboxes = []

        # It then loops through all the blobs it found
        for blob in blobs:
            # For each blob it checks to make sure the amount of pixels
            # in the blobs is greater than a certain threshhold so that
            # random pixels dont get recognized as people
            if len(blob) > 30:
                # It then converts the blob to an array
                blob = np.array(blob)
                blob = blob.reshape((blob.shape[0], 2))

                averageBbox = np.array([0.0, 0.0, 0.0, 0.0])

                # This loops through all the pixels in the blob
                for i in range (0, len(blob)):
                    # It then gets the bounding box in that pixel
                    currentBbox = modelOutput[1:, blob[i][1], blob[i][0]]

                    # And adds it to the average bounding box
                    averageBbox[0] += blob[i][0]/256*1024 + currentBbox[0]*1024
                    averageBbox[1] += blob[i][1]/256*1024 + currentBbox[1]*1024
                    averageBbox[2] += currentBbox[2]*1024
                    averageBbox[3] += currentBbox[3]*1024

                # To get the average the average bounding box is divided
                # by the amount of pixels in the blob
                averageBbox = (averageBbox/len(blob)).astype(int)
                # And it appends the bounding box
                bboxes.append(averageBbox)

        # This is a new list of bounding boxes that will delete any
        # overlaping bounding boxes
        filtered_bboxes = []
        # This loops through all of the bboxes
        for bbox in bboxes:
            # First it gets the top left coord and the bottom right coord
            # of the bounding box
            x = bbox[0]
            y = bbox[1]
            x1 = bbox[0] + bbox[2]
            y1 = bbox[1] + bbox[3]
            
            # If the length of the filtered bounding boxes is 0 it just adds the curent coords
            if len(filtered_bboxes) == 0:
                filtered_bboxes.append([x, y, x1, y1])
            else:
                # Otherwise it checks for overlap
                overlaping = False
                # This loops through the filtered bounding boxes
                for i in range (0, len(filtered_bboxes)):
                    # It gets the overlap percentage of the current bounging box and
                    # the one it is being compared to
                    overlaping_area = rectangle_overlap(x, y, x1, y1, filtered_bboxes[i][0], filtered_bboxes[i][1], filtered_bboxes[i][2], filtered_bboxes[i][3])

                    # If the overlap percentage is greater than a certain threshhold
                    # it sets overlaping to true
                    if overlaping_area > 0.1:
                        overlaping = True

                # It only adds the bounding box if overlaping is not true
                if overlaping == False:
                    filtered_bboxes.append([x, y, x1, y1])

        # Then it draws all of the bounding boxes on the image
        for bbox in filtered_bboxes:
            outputImage = cv2.rectangle(outputImage, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color=(0, 0, 255), thickness=2)

    # And finally it displays the image
    cv2.imshow("video", outputImage)
    cv2.waitKey(1)
