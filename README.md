# Person_Detection
 
![PersonDetectorNet in action](demo_person_detector_net.jpg)

### Overview:
This is a convolutional neural network that can detect people. Teh CNN has 13 convolutional layers and 8 deconvolutional layers. It takes in an RGB image and ouputs a tensor with 5 channels per pixel (person_value, bbox x, bbox y, bbox width, bbox height). The ouput tensor is then input into the post proccessing. The post proccessing algorithm first finds blobs of pixels in the output mask which is the first channel. It then gets the average bounding box per blob. And finally it deletes any overlaping bounding boxes to prevent there being more than one bounding box per person.

In this project I wanted get data about how many people are in a room at my school and at what time. To do this I got this neural network to run on a Jetson Nano that used a camera to get live video footage of the room. I then calculated the number of people in the room and using the neural network, and sent the data to the cloud using http. I graphed the results which can be found in the Visualizations folder.

#### Dataset
To train the nerual network I used the CrowdHuman dataset. It contains 15,000 annotated images with bounding boxes for each person in the image.

@article{shao2018crowdhuman,
    title={CrowdHuman: A Benchmark for Detecting Human in a Crowd},
    author={Shao, Shuai and Zhao, Zijian and Li, Boxun and Xiao, Tete and Yu, Gang and Zhang, Xiangyu and Sun, Jian},
    journal={arXiv preprint arXiv:1805.00123},
    year={2018}
  }
