# This program contains a function that parses the data

import pandas as pd
import json
import torch
import numpy as np

# This function parses the data
# path_to_annotations is the path that the datasets annotations are stored in
def parse_data(path_to_annotations, max_people_count):
    # First it opens the file it is in .odgt format
    # This is means that it is a json stored in a text file
    # So to read it I need to read it in as a text file and then convert
    # the text to a json file
    with open(path_to_annotations) as f:
        print("loading data")
        # This creates a list that will store the data
        data = []
        
        # This is a counter which isnt used for anything except for printing
        # it out to see the progress
        counter = 0
        # This loops through all the lines
        for line in f:
            # It tries to load the current line as a json
            # if it rhows an error it breaks
            try:
                jsonData = json.loads(line)
            except:
                break

            # This converts the json data to a pandas data frame
            image = pd.DataFrame(jsonData)

            # It then gets the file name the image from the dataframe
            # Since each line is a json for a single image I use an index of 0 for that image
            filename = image["ID"][0]
            
            # This creates an empty tesor for the bounding boxes in the image
            # There is a max of 50 people and the 4 values for each is x, y, width, height of the bounding box
            people = torch.zeros((max_people_count*4))

            # This loops through each person
            for i in range (0, max_people_count):
                # It tries to add the data
                # If it fails that means that there are less than the max number of people
                # If it fails it just adds all zeros for that person
                try:
                    # This sets the x, y, width, and height
                    people[i*4] = image["gtboxes"][i]["fbox"][0]
                    people[i*4 + 1] = image["gtboxes"][i]["fbox"][1]
                    people[i*4 + 2] = image["gtboxes"][i]["fbox"][2]
                    people[i*4 + 3] = image["gtboxes"][i]["fbox"][3]
                except:
                    # This sets everything to 0
                    people[i*4] = 0.0
                    people[i*4 + 1] = 0.0
                    people[i*4 + 2] = 0.0
                    people[i*4 + 3] = 0.0
                
            # This appends the filename and the bounding boxes as a list
            data.append([filename, people])
            # It then increments and prints out the counter
            counter += 1
            print(counter)

        print("Finished loading data")
        # Finally it converts the data to an array and returns it
        return np.array(data, dtype=object)

    
#data = parse_data("data/annotation_train.odgt")