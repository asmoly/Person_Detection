import pandas as pd
import json
import torch
import numpy as np
import cv2

from PIL import Image

def parse_data(path_to_annotations):
    with open(path_to_annotations) as f:
        print("loading data")
        data = []
        
        counter = 0
        for line in f:
            try:
                jsonData = json.loads(line)
            except:
                break

            image = pd.DataFrame(jsonData)

            filename = image["ID"][0]
            
            number_of_people = image["gtboxes"].shape[0]
            people = torch.zeros((50*4))

            for i in range (0, 50):
                bounding_box = 0
                try:
                    #bounding_box = torch.Tensor(image["gtboxes"][i]["fbox"])
                    people[i*4] = image["gtboxes"][i]["fbox"][0]
                    people[i*4 + 1] = image["gtboxes"][i]["fbox"][1]
                    people[i*4 + 2] = image["gtboxes"][i]["fbox"][2]
                    people[i*4 + 3] = image["gtboxes"][i]["fbox"][3]
                except:
                    #bounding_box = torch.Tensor([0.0, 0.0, 0.0, 0.0])
                    people[i*4] = 0.0
                    people[i*4 + 1] = 0.0
                    people[i*4 + 2] = 0.0
                    people[i*4 + 3] = 0.0
                
                #people[i] = bounding_box

            data.append([filename, people])
            counter += 1
            print(counter)

        print("Finished loading data")
        return np.array(data, dtype=object)

        

#data = parse_data("data/annotation_train.odgt")