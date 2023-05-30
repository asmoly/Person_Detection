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

def rectangle_overlap(ax, ay, ax1, ay1, bx, by, bx1, by1):
    dx = min(ax1, bx1) - max(ax, bx)
    dy = min(ay1, by1) - max(ay, by)

    rectangle1_area = (ax1 - ax)*(ay1 - ay)

    overlaping_area = 0
    if dx >= 0 and dy >= 0:
        overlaping_area =  dx*dy
    
    return overlaping_area/rectangle1_area

print("Starting person counter")

print("Opening camera 0")
cam = cv2.VideoCapture(CAMERA, cv2.CAP_DSHOW)
cam.set(cv2.CAP_PROP_FRAME_WIDTH, 1024)
cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 1024)
print("Opened camera")

print("Opening device")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Loaded device", device)

print("Loading model")
model = person_finder_AI.load_model("personnet_72.pt").to(device)
print("Loaded model")

while True:
    ret, image = cam.read()

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # image is RGB now
    image = Image.fromarray(image)
    image = image.crop((0, 0, 1024, 1024))
    image = transforms.ToTensor()(image)

    

    with torch.no_grad():
        modelOutput = model(image.to(device)).cpu()
        modelOutput[0] = nn.Threshold(0.98, 0.0)(modelOutput[0])

        # Seperating mask from model output and resising it
        outputMask = modelOutput[0:1]
        outputMask = fn.resize(transforms.ToPILImage()(outputMask), size=[1024])
        outputMask = transforms.ToTensor()(outputMask)

        # This is for if you want to see the mask
        # cv2.imshow("test", (outputMask*255).permute(1, 2, 0).numpy().astype(np.uint8))
        # cv2.waitKey(1)

        # Drawing mask on original image
        outputImage = torch.Tensor(3, 1024, 1024)
        outputImage[0:3] = (image*255.0)[0:3]   # image is CHW
        outputImage[1:2] += outputMask*100.0           # outputImage is CHW with RGBA
        
        # Formating output image
        outputImage = torch.clamp(outputImage, 0.0, 255.0)
        outputImage = outputImage.permute(1, 2, 0)  # outputImage is HWC with RGBA
        outputImage = np.asarray(outputImage).astype(np.uint8).copy() 
        outputImage = cv2.cvtColor(outputImage, cv2.COLOR_RGB2BGR)


        resultMask = (modelOutput[0]*255).cpu().detach().numpy().astype(np.uint8)
        mask = np.zeros(image.shape, dtype=np.uint8)
        thresh = cv2.threshold(resultMask, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

        blobs = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        blobs = blobs[0] if len(blobs) == 2 else blobs[1]

        bboxes = []
        for blob in blobs:
            if len(blob) > 30:
                blob = np.array(blob)
                blob = blob.reshape((blob.shape[0], 2))

                averageBbox = np.array([0.0, 0.0, 0.0, 0.0])

                for i in range (0, len(blob)):
                    currentBbox = modelOutput[1:, blob[i][1], blob[i][0]]

                    averageBbox[0] += blob[i][0]/256*1024 + currentBbox[0]*1024
                    averageBbox[1] += blob[i][1]/256*1024 + currentBbox[1]*1024
                    averageBbox[2] += currentBbox[2]*1024
                    averageBbox[3] += currentBbox[3]*1024

                averageBbox = (averageBbox/len(blob)).astype(int)
                bboxes.append(averageBbox)

                #outputImage = cv2.rectangle(outputImage, (averageBbox[0], averageBbox[1]), (averageBbox[0] + averageBbox[2], averageBbox[1] + averageBbox[3]), color=(0, 0, 255), thickness=2)

        new_bboxes = []
        for bbox in bboxes:
            x = bbox[0]
            y = bbox[1]
            x1 = bbox[0] + bbox[2]
            y1 = bbox[1] + bbox[3]
            
            if len(new_bboxes) == 0:
                new_bboxes.append([x, y, x1, y1])
            else:
                overlaping = False
                for i in range (0, len(new_bboxes)):
                    overlaping_area = rectangle_overlap(x, y, x1, y1, new_bboxes[i][0], new_bboxes[i][1], new_bboxes[i][2], new_bboxes[i][3])

                    if overlaping_area > 0.1:
                        overlaping = True

                if overlaping == False:
                    new_bboxes.append([x, y, x1, y1])

        for bbox in new_bboxes:
            outputImage = cv2.rectangle(outputImage, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color=(0, 0, 255), thickness=2)

    cv2.imshow("video", outputImage)
    cv2.waitKey(1)
