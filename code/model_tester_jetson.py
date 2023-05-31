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

# This is the camera device to get video from
CAM_DEVICE = "/dev/video0"
# Resolution of camera
CAM_WIDTH = 640
CAM_HEIGHT = 480
# File with neural network model
MODEL_FILE = "personnet_72.pt"
# Input and ouput resolution of the neural network
NN_IN_WIDTH = 1024
NN_IN_HEIGHT = 1024
NN_OUT_WIDTH = 256
NN_OUT_HEIGHT = 256

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

# This is the main function that runs the main loop
def main():
    print("Starting person counter")

    print("Opening camera {}".format(CAM_DEVICE))
    # First it opens the camera using the correct path
    # It uses linux's v4l2 camera driver
    cam = cv2.VideoCapture(CAM_DEVICE, cv2.CAP_V4L2)
    # It also sets the resolution
    cam.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_WIDTH)
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_HEIGHT)
    
    # Then checks if the camera opened successfully
    if cam.isOpened():
        print("Opened camera")
    else:
        print("Cannot open camera {}".format(CAM_DEVICE))
        return

    print("Opening device")
    # This sets the device, if the GPU is available it will use that
    # otherwise it uses the CPU
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Opened device", device)

    print("Loading model from {}".format("personnet_72.pt"))
    # This loads the model from the specified path
    model = person_finder_AI.load_model(MODEL_FILE).to(device)
    print("Loaded model")

    # this creates an empty tensor for the model output later
    modelOutput = torch.empty((1, NN_OUT_HEIGHT, NN_OUT_WIDTH), dtype=torch.float)
    while True:
        # This gets the current frame
        ret, image = cam.read()
        print(image.shape)

        # First it converts the image to RGB
        frame = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # image is RGB now
        # Then it converts it to pillow image, crops it, and resizes it
        image = Image.fromarray(frame)
        image = image.crop((0, 0, CAM_HEIGHT, CAM_HEIGHT))
        image = image.resize((NN_IN_WIDTH, NN_IN_HEIGHT))
        # And it converts it to tensor
        image_tensor = transforms.ToTensor()(image)

        print(image_tensor.shape)

        # Then it calculates the model output without calculating the gradient
        with torch.no_grad():
            # This makes the input be a batch with on element
            model_input = torch.empty((1, 3, NN_IN_HEIGHT, NN_IN_WIDTH))
            model_input[0] = image_tensor	        
            modelOutput = model(model_input.to(device)).cpu()
		
        print("model out shape:", modelOutput.shape)
        # This gets rid of the batch
        modelOutput = modelOutput[0]
        # And it threshholds the first channel
        modelOutput[0] = nn.Threshold(0.98, 0.0)(modelOutput[0])

        # Separating mask from model output and resizing it
        outputMask = modelOutput[0:1]
        outputMask = fn.resize(transforms.ToPILImage()(outputMask), size=[1024])
        outputMask = transforms.ToTensor()(outputMask)

        # This is for if you want to see the mask
        #cv2.imshow("mask", (outputMask*255).permute(1, 2, 0).numpy().astype(np.uint8))
        #cv2.waitKey(0)

        # Drawing mask on original image
        outputImage = torch.Tensor(3, 1024, 1024)
        # Adds the original image and making it from 0 to 255
        outputImage[0:3] = (image_tensor*255.0)[0:3]
        # And it adds the ouput mask to the green channel so that the
        # ouput mask looks green on the image
        outputImage[1:2] += outputMask*100.0

        # Formatting output image
        outputImage = torch.clamp(outputImage, 0.0, 255.0)
        # Converts it to Height width channel
        outputImage = outputImage.permute(1, 2, 0)  # outputImage is HWC with RGBA
        # Converts it to array and to BGR
        outputImage = np.asarray(outputImage).astype(np.uint8).copy() 
        outputImage = cv2.cvtColor(outputImage, cv2.COLOR_RGB2BGR)

        # This bring the ouput to 0 - 255
        resultMask = (modelOutput[0]*255).cpu().detach().numpy().astype(np.uint8)
        mask = np.zeros(image_tensor.shape, dtype=np.uint8)
        thresh = cv2.threshold(resultMask, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

        # This finds contours in order to get all the blobs of pixels in
        # the output mask
        blobs = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        blobs = blobs[0] if len(blobs) == 2 else blobs[1]

        bboxes = []
        # This loops through the blobs
        for blob in blobs:
            # Then it checks if the number of pixels is greater than a specified
            # number to filter out random pixels
            if len(blob) > 10:
                # Then it converts the blob into a numpy array
                blob = np.array(blob)
                blob = blob.reshape((blob.shape[0], 2))

                averageBbox = np.array([0.0, 0.0, 0.0, 0.0])

                # This loops through all of the pixels in the blob
                for i in range (0, len(blob)):
                    # This gets the x, y, width, height for the current bounding box
                    currentBbox = modelOutput[1:, blob[i][1], blob[i][0]]

                    # Then it adds the curent bounding box and scales it
                    averageBbox[0] += blob[i][0]/NN_OUT_WIDTH*NN_IN_WIDTH + currentBbox[0]*NN_IN_WIDTH
                    averageBbox[1] += blob[i][1]/NN_OUT_HEIGHT*NN_IN_HEIGHT + currentBbox[1]*NN_IN_HEIGHT
                    averageBbox[2] += currentBbox[2]*NN_IN_WIDTH
                    averageBbox[3] += currentBbox[3]*NN_IN_HEIGHT

                # It then divides the bounding box by the number of pixels to get
                # an average bounding box
                averageBbox = (averageBbox/len(blob)).astype(int)
                # And it adds the bounding box to the list
                bboxes.append(averageBbox)

        # This is a list of filtered bounding boxes which will delete
        # any overlaping bounding boxes
        filtered_bboxes = []
        # This loops through all the bounding boxes
        for bbox in bboxes:
            # This gets the top left coord and the bottom right coord
            x = bbox[0]
            y = bbox[1]
            x1 = bbox[0] + bbox[2]
            y1 = bbox[1] + bbox[3]

            # First it check if there are any boxes in the list
            if len(filtered_bboxes) == 0:
                # If there are not then it adds the current bounding box
                filtered_bboxes.append([x, y, x1, y1])
            else:
                overlaping = False
                # Otherwise it loops through the filtered bounding boxes
                # to check for overlap
                for i in range (0, len(filtered_bboxes)):
                    # This gets the overlaping percentage
                    overlaping_area = rectangle_overlap(x, y, x1, y1, filtered_bboxes[i][0], filtered_bboxes[i][1], filtered_bboxes[i][2], filtered_bboxes[i][3])

                    # It then checks if the overlap is greater than some percent
                    if overlaping_area > 0.5:
                        # If it is overlaping it sets this to true
                        overlaping = True

                    # It only adds to the list if the bounding box doesnt
                    # overlap with anything
                    if overlaping == False:
                        filtered_bboxes.append([x, y, x1, y1])

        # Then it draws all the bounding boxes
        for bbox in filtered_bboxes:
            outputImage = cv2.rectangle(outputImage, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color=(0, 0, 255), thickness=2)

        # And finally it displays the image
        cv2.imshow("video", outputImage)
        cv2.waitKey(1)


if __name__ == "__main__":
    main()

