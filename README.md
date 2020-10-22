# Single Object Detector

# INTRODUCTION
Object detection is a very mature technique, and a lot of neural networks are used to detect objects in images, for example, RCNN, YOLO, SSD, MASK-RCNN. While there are lot of reopsitories for the object detector networks, but to train these networks on a customized dataset, a good amount of effort is required for generating the dataset, as well as converting and store the dataset in the appropriate location. This project aims to design a pipeline with a user-friendly interface for generating the dataset and training the network for a customized dataset. 

# INTERFACE
To up the interface, type in the terminal `cd /path/to/the/repository` and `python main_file.py`. The Current frame will be dispalyed with some virtual buttons as shown below.


<p align="center">
  <img src="/interface_images/img_interface.png" />
</p>

Each Button has a specific function, for example, 
1. `Ext` exit the code.
2. `Sav` save the current frame. By default images will be saved in folder `<./save_images/set1>`. To save in different folder like `<./save_images/set2>`, copy the folder `./save_images/empty_set` and rename it to `<./save_images/set2>` and set the optional argument `-s 2` for the command `python main_file.py`.
3. `Tmr` save the current frame after `5` seconds. Default value is `5 secs`, but can be modify by setting the optional argument `-t 3` for the command `python main_file.py`.
4. `Lab` Annotate the saved images. Path of the saved images is decided by the `-s` optional argument.
5. `All` visualize the all annotated images. Further when `All` is selected, the annotated image with some virtual buttons will appear as shown below.

<p align="center">
  <img src="/interface_images/All_img_interface.png" />
</p>


where, `>` represents next annnotated image, `<` represents previous annotated image and `Mod` represents modify the bounding box.


# Button Selection
To select any button, either touch the button with your palm or put the cursor on the button and press left click. For predicting the palm locations [posenet](https://github.com/rwightman/posenet-pytorch) is used and red circles are drawn at the palm position.


# Training
For training, use the command `python training.py` with optional arguments

1. `-i `, number of iterations, default 1000.
2. `-b `, batch size, default 2.
3. `-s `, set number, default 1.


# Test
For testing, use the command `python test_web_cam.py`. Demos Links are [Set1 Demo](https://drive.google.com/file/d/1erjrL7dJ3ZSebuqrm9VctombHVDqi8e7/view?usp=sharing), [Set2 Demo](https://www.youtube.com/watch?v=TxFR06irYeE).








**TO DO**
1. Add the filter for smooth output.


