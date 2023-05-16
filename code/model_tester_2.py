import cv2
import numpy as np
from PIL import Image

import torch
import torchvision.transforms as transforms

import person_finder_AI

IN_IMAGE_SIZE = 1024


cam = cv2.VideoCapture(1, cv2.CAP_DSHOW)
cam.set(cv2.CAP_PROP_FRAME_WIDTH, 1024)
cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 1024)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = person_finder_AI.load_model("D:/Person_Finder_Logs/facenet_64.pt").to(device)

while True:
    ret, image = cam.read()
    outputImage = image

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # image is RGB now
    image = Image.fromarray(image)
    image = image.crop((0, 0, 1024, 1024))
    image = transforms.ToTensor()(image)

    #outputImage = image.permute(1, 2, 0)  # outputImage is HWC with RGBA
    #outputImage = np.asarray(outputImage).astype(np.uint8).copy() 
    #outputImage = cv2.cvtColor(outputImage, cv2.COLOR_RGB2BGR)

    with torch.no_grad():
        model_output = model(image.to(device)).cpu()
        model_output[0] = torch.nn.Threshold(0.95, 0.0)(model_output[0])

        non_zero_values = torch.nonzero(model_output[0])
        
        bboxes = []
        counter = 0
        while counter < non_zero_values.shape[0]:
            pixel_index = non_zero_values[counter]

            x = pixel_index[1]/256*IN_IMAGE_SIZE
            y = pixel_index[0]/256*IN_IMAGE_SIZE
            
            width = model_output[3, pixel_index[0], pixel_index[1]]*IN_IMAGE_SIZE
            height = model_output[4, pixel_index[0], pixel_index[1]]*IN_IMAGE_SIZE

            if len(bboxes) == 0:
                bboxes.append([x, y, x + width, y + height])
            else:
                for i in range (0, len(bboxes)):
                    dx = min(int(bboxes[i][2]), int(x + width)) - max(int(bboxes[i][0]), int(x))
                    dy = min(int(bboxes[i][3]), int(y + height)) - max(int(bboxes[i][1]), int(y))
                    
                    if dx >= 0 and dy >= 0:
                        overlap_area = dx*dy
                    else:
                        overlap_area = 0

                    if overlap_area < 1:
                        bboxes.append([x, y, x + width, y + height])

            counter += 300

        for bbox in bboxes:
            cv2.rectangle(outputImage, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), color=(0, 0, 255), thickness=2)

    cv2.imshow("model_tester", outputImage)
    cv2.waitKey(1)
        