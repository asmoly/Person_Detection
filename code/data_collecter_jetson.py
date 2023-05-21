import cv2
import numpy as np
import requests
import time
import threading
from PIL import Image

import torchvision.transforms.functional as fn
import torch
import torchvision.transforms as transforms
import torch.nn as nn

import person_finder_AI

CAM_DEVICE = "/dev/video0"
CAM_WIDTH = 640
CAM_HEIGHT = 480
MODEL_FILE = "personnet_72.pt"

NN_IN_WIDTH = 1024
NN_IN_HEIGHT = 1024
NN_OUT_WIDTH = 256
NN_OUT_HEIGHT = 256

people_count_list = []

def timer(time_interval, log_name):
    time = 0

    while True:
        if time >= time_interval:
            average_people_count = 0
            for people_count in people_count_list:
                average_people_count += people_count

            average_people_count = average_people_count/len(people_count_list)
            print(f"Average count of people is: {average_people_count}")

            data = data_logger_string("sasha", log_name, average_people_count)

            sent_data = False
            while sent_data == False:
                try:
                    requests.post(data)
                    sent_data = True
                    print("Succesfully sent data")
                except:
                    sent_data = False
                    print("Could not send data")

            time = 0
            people_count_list = []

        time.sleep(1)
        time += 1

def data_logger_string(username, device_id, people_count):
    # It then creates the string where the first part is the link to the server
    # and then it adds all the parameter values in a specific formatt
    datalogger_url_str = "https://eps-datalogger.herokuapp.com/api/data/" \
    + username + "/add?device_id=" + device_id + "&int1=" + str(people_count)

    return datalogger_url_str

def rectangle_overlap(ax, ay, ax1, ay1, bx, by, bx1, by1):
    dx = min(ax1, bx1) - max(ax, bx)
    dy = min(ay1, by1) - max(ay, by)

    rectangle1_area = (ax1 - ax)*(ay1 - ay)

    overlaping_area = 0
    if dx >= 0 and dy >= 0:
        overlaping_area =  dx*dy
    
    return overlaping_area/rectangle1_area

def get_bboxes(blobs, modelOutput, pixel_thresh):
    bboxes = []
    for blob in blobs:
        if len(blob) > pixel_thresh:
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

    return bboxes

def filter_bboxes(bboxes, overlap_thresh):
    filtered_bboxes = []
    for bbox in bboxes:
        x = bbox[0]
        y = bbox[1]
        x1 = bbox[0] + bbox[2]
        y1 = bbox[1] + bbox[3]

        if len(filtered_bboxes) == 0:
            filtered_bboxes.append([x, y, x1, y1])
        else:
            overlaping = False
            for i in range (0, len(filtered_bboxes)):
                overlaping_area = rectangle_overlap(x, y, x1, y1, filtered_bboxes[i][0], filtered_bboxes[i][1], filtered_bboxes[i][2], filtered_bboxes[i][3])

                if overlaping_area > overlap_thresh:
                    overlaping = True

            if overlaping == False:
                filtered_bboxes.append([x, y, x1, y1])

    return filtered_bboxes

def main():
    print("Starting person counter")

    print("Opening camera {}".format(CAM_DEVICE))
    cam = cv2.VideoCapture(CAM_DEVICE, cv2.CAP_V4L2)
    cam.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_WIDTH)
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_HEIGHT)
    if cam.isOpened():
        print("Opened camera")
    else:
        print("Cannot open camera {}".format(CAM_DEVICE))
        return

    print("Opening device")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Opened device:", device)

    print("Loading model from {}".format("personnet_72.pt"))
    model = person_finder_AI.load_model(MODEL_FILE).to(device)
    print("Loaded model")

    modelOutput = torch.empty((1, NN_OUT_HEIGHT, NN_OUT_WIDTH), dtype=torch.float)
    while True:
        ret, image = cam.read()

        frame = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # image is RGB now
        image = Image.fromarray(frame)
        image = image.crop((0, 0, CAM_HEIGHT, CAM_HEIGHT))
        image = image.resize((NN_IN_WIDTH, NN_IN_HEIGHT))
        image_tensor = transforms.ToTensor()(image)

        with torch.no_grad():
            model_input = torch.empty((1, 3, NN_IN_HEIGHT, NN_IN_WIDTH))
            model_input[0] = image_tensor	        
            modelOutput = model(model_input.to(device)).cpu()
		
        modelOutput = modelOutput[0]
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
        outputImage[0:3] = (image_tensor*255.0)[0:3]   # image is CHW
        outputImage[1:2] += outputMask*100.0           # outputImage is CHW with RGBA

        # Formatting output image
        outputImage = torch.clamp(outputImage, 0.0, 255.0)
        outputImage = outputImage.permute(1, 2, 0)  # outputImage is HWC with RGBA
        outputImage = np.asarray(outputImage).astype(np.uint8).copy() 
        outputImage = cv2.cvtColor(outputImage, cv2.COLOR_RGB2BGR)

        resultMask = (modelOutput[0]*255).cpu().detach().numpy().astype(np.uint8)
        thresh = cv2.threshold(resultMask, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

        blobs = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        blobs = blobs[0] if len(blobs) == 2 else blobs[1]

        bboxes = get_bboxes(blobs, modelOutput, 30)
        filtered_bboxes = filter_bboxes(bboxes, 50)

        people_count = len(filtered_bboxes)
        for bbox in filtered_bboxes:
            outputImage = cv2.rectangle(outputImage, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color=(0, 0, 255), thickness=2)

        people_count_list.append(people_count)

        cv2.imshow("video", outputImage)
        cv2.waitKey(1)
        
        time.sleep(20)


if __name__ == "__main__":
    threading.Thread(target=timer, args=(600, "person_counter_test",), daemon=True).start()
    main()

