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

# This is the name of the camera, since it is on jetson
# it is stored in a pth
CAM_DEVICE = "/dev/video0"
# This is the input width and height of the camera
CAM_WIDTH = 640
CAM_HEIGHT = 480
# This is the file that the model is stored in
MODEL_FILE = "personnet_72.pt"

# This is the input resolution to the neural network
NN_IN_WIDTH = 1024
NN_IN_HEIGHT = 1024

# This is the ouput resolution of the neural network
NN_OUT_WIDTH = 256
NN_OUT_HEIGHT = 256

# These are the variables to calculate the sum over the specified time
# people_count_sum is the cumilative count of all the people found over that time
# people_count_counter is just the amount of times it found people
# Then it divides the sum by the counter to get the average
people_count_sum = 0
people_count_counter = 0

# Timer runs on a seperate thread
# It runs a timer and after a specified time has elapsed it send data to the EPS servers
def timer(time_interval, log_name):
    global people_count_sum
    global people_count_counter

    # This variable is used to track the time
    time = 0

    # This is while true loop that runs forever until the program is terminated
    while True:
        # This runs if the timer has reached the specified time
        if time >= time_interval:
            # First it calculates the average number of people over the time period
            average_people_count = people_count_sum/people_count_counter
            # Then it resets the sum and counter variables
            people_count_sum = 0
            people_count_counter = 0

            # It prints out the average number of people for debugging
            print(f"Average count of people is: {average_people_count}")

            # Then it uses the data logger string to generate a link with the data to post
            data = data_logger_string("sasha", log_name, average_people_count)

            # This variable is used to check if the data was sent
            sent_data = False
            # This loop will run until the data is sent
            while sent_data == False:
                # First it tried to post the data
                try:
                    requests.post(data)
                    # If it sent then it sets the sent_data variable to true which exits the loop
                    sent_data = True
                    print("Succesfully sent data")
                except:
                    # Otherwise it sets it to false meaning the loop will continue
                    # until the data is posted successfully
                    sent_data = False
                    print("Could not send data")

            # And it resets the timer
            time = 0

        # This sleep one second and adds one to the timer to keep track of time correctly
        time.sleep(1)
        time += 1

# This functions takes in the username, device_id, and the people_count and creates a link
# to send. The username and device id are used to make links unique
# and the people count gets inputed into float1 to get posted
def data_logger_string(username, device_id, people_count):
    # It then creates the string where the first part is the link to the server
    # and then it adds all the parameter values in a specific formatt
    datalogger_url_str = "https://eps-datalogger.herokuapp.com/api/data/" \
    + username + "/add?device_id=" + device_id + "&int1=" + str(people_count)

    return datalogger_url_str

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

# This function takes in the modelOutput which is what the neural network outputs
# the blobs which are the blobs of pixels and the threshhold for the number of pixels in each bounding box
# And it can draw bounding boxes for each blob
def get_bboxes(blobs, modelOutput, pixel_thresh):
    # First it makes a list for the bounding boxes
    bboxes = []
    # Then it loops through each of the blobs
    for blob in blobs:
        # It first checks that the blob has more than a certain amount
        # of pixels
        # This is done to filter out random blobs that arn't on a person
        if len(blob) > pixel_thresh:
            # It then converts the blob to a numpy array
            blob = np.array(blob)
            # And reshapes it
            blob = blob.reshape((blob.shape[0], 2))

            # It then creates an empy array for the bounding box
            averageBbox = np.array([0.0, 0.0, 0.0, 0.0])

            # This loops through all the pixels of the blob
            for i in range (0, len(blob)):
                # It then gets the current bounding box from the values of the pixel
                # which the neural network outputed
                currentBbox = modelOutput[1:, blob[i][1], blob[i][0]]

                # It then normalizes and scales the bounding box and adds it to the average boudnign box
                averageBbox[0] += blob[i][0]/NN_OUT_WIDTH*NN_IN_WIDTH + currentBbox[0]*NN_IN_WIDTH
                averageBbox[1] += blob[i][1]/NN_OUT_HEIGHT*NN_IN_HEIGHT + currentBbox[1]*NN_IN_HEIGHT
                averageBbox[2] += currentBbox[2]*NN_IN_WIDTH
                averageBbox[3] += currentBbox[3]*NN_IN_HEIGHT

            # It then averages the bounding box by dividing by the number of pixel sin the blob
            averageBbox = (averageBbox/len(blob)).astype(int)
            # And it appends it the bboxes list
            bboxes.append(averageBbox)

    return bboxes

# This function deletes bounding box that are overlaping
# It takes in a list of bounding box to filter
# And the percentage of overlap to delete a bounding box
def filter_bboxes(bboxes, overlap_thresh):
    # This creates a list for the filtered bounding boxes
    filtered_bboxes = []
    # This loops through the bounding boxes
    for bbox in bboxes:
        # This creates variables for the top left coord and bottom right coord
        x = bbox[0]
        y = bbox[1]
        x1 = bbox[0] + bbox[2]
        y1 = bbox[1] + bbox[3]

        # If there are no bounding boxes in the list then it just adds the current bounding box
        if len(filtered_bboxes) == 0:
            filtered_bboxes.append([x, y, x1, y1])
        else:
            # Otherwise it checks for overlaping rectangles
            overlaping = False
            # This loops through the filtered bounding boxes
            for i in range (0, len(filtered_bboxes)):
                # It calculates the percentage overlaping between the current rectangle
                # and the one in the filtered bounding boxes
                overlaping_area = rectangle_overlap(x, y, x1, y1, filtered_bboxes[i][0], filtered_bboxes[i][1], filtered_bboxes[i][2], filtered_bboxes[i][3])

                # If the percentage is greater than a specified value
                # it sets overlaping to true
                if overlaping_area > overlap_thresh:
                    overlaping = True

            # Then it only adds the current bounding box to the filtered
            # list if it is not overlaping with anything
            if overlaping == False:
                filtered_bboxes.append([x, y, x1, y1])

    return filtered_bboxes

# This is the main function that runs all the functions
def main():
    global people_count_sum
    global people_count_counter

    print("Starting person counter")

    # First it opens the camera
    print("Opening camera {}".format(CAM_DEVICE))
    # It uses the v4l2 driver which is linux's camera driver
    cam = cv2.VideoCapture(CAM_DEVICE, cv2.CAP_V4L2)
    # It also sets the resolution of the camera
    cam.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_WIDTH)
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_HEIGHT)
    
    # Then it checks if the camera is open
    if cam.isOpened():
        print("Opened camera")
    else:
        print("Cannot open camera {}".format(CAM_DEVICE))
        return

    print("Opening device")
    # This creates the device which will try find a GPU and if it doesnt
    # it will use the CPU
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Opened device:", device)

    print("Loading model from {}".format("personnet_72.pt"))
    # It then loads the model file and converts it to the GPU
    model = person_finder_AI.load_model(MODEL_FILE).to(device)
    print("Loaded model")

    # It then creates an empty tensor for the model_ouput
    modelOutput = torch.empty((1, NN_OUT_HEIGHT, NN_OUT_WIDTH), dtype=torch.float)
    while True:
        # It then reads in the image from the camera
        ret, image = cam.read()

        # It converts the color to RGB
        frame = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # image is RGB now
        image = Image.fromarray(frame)
        # It them crops the image to the resolution
        image = image.crop((0, 0, CAM_HEIGHT, CAM_HEIGHT))
        # Since the camera resolution is smaller than the neural network's input
        # it resizes the camera image to the desired resolution
        image = image.resize((NN_IN_WIDTH, NN_IN_HEIGHT))
        # And then it transforms the image to a tensor
        image_tensor = transforms.ToTensor()(image)

        # torch.no_grad makes sure that when it runs the model it doesnt calculate
        # the gradients which makes it run a lot faster
        with torch.no_grad():
            # It then turns the input image into a batch with only one element
            # because jetson pytorch version is older and this is required
            model_input = torch.empty((1, 3, NN_IN_HEIGHT, NN_IN_WIDTH))
            model_input[0] = image_tensor	  
            # Then it runs the model with the camera image and converts it back to the cpu
            # for post proccessing       
            modelOutput = model(model_input.to(device)).cpu()
		
        # This gets rid of the extra batch
        modelOutput = modelOutput[0]
        # This threshholds it so that all pixels that dont have a value of 0.98 get brought to 0
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
        # Clamps the rgb values to be between 0 and 255
        outputImage = torch.clamp(outputImage, 0.0, 255.0)
        # Converts it height width channel
        outputImage = outputImage.permute(1, 2, 0)
        # Converts it to array
        outputImage = np.asarray(outputImage).astype(np.uint8).copy() 
        # Converts it from rgb to bgr
        outputImage = cv2.cvtColor(outputImage, cv2.COLOR_RGB2BGR)

        # Gets the mask from the model output and converts it to array
        resultMask = (modelOutput[0]*255).cpu().detach().numpy().astype(np.uint8)
        thresh = cv2.threshold(resultMask, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

        # This finds contours in order to find blobs of pixels in the mask
        blobs = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # This finads all of the blobs and gets rid of batch
        blobs = blobs[0] if len(blobs) == 2 else blobs[1]

        # It then uses the get_bboxes function to get a bounding box
        # for each of the blobs
        bboxes = get_bboxes(blobs, modelOutput, 30)
        # And then it filters all of te overlaping bounding boxes
        filtered_bboxes = filter_bboxes(bboxes, 0.5)

        # This get the number of people by just counting the number of bounding boxes
        people_count = len(filtered_bboxes)
        # It then loops through the bounding boxes and draws all of them on the image
        for bbox in filtered_bboxes:
            outputImage = cv2.rectangle(outputImage, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color=(0, 0, 255), thickness=2)

        # It also adds the people count to the sum and increments the counter
        people_count_sum += people_count
        people_count_counter += 1

        # It then displays the image with bounding boxes for debugging
        cv2.imshow("video", outputImage)
        cv2.waitKey(1)
        
        # It only runs this every minute because without this the jetson would overheat
        time.sleep(60)

# This runs botht the thread and the main function
if __name__ == "__main__":
    # This Runs the timer on a seperate thread so that the timer is accurate and it can
    # run at the same time as the main function
    threading.Thread(target=timer, args=(600, "person_counter_test",), daemon=True).start()
    # This runs the main function
    main()

