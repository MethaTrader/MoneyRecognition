# Real-time banknote recognition
This project was created for training purposes to implement photo or video recognition of Ukrainian banknotes.
![example](https://s9.gifyu.com/images/ezgif-1-30f01ba8d5.gif)

# Libraries/frameworks used

We used **Python** programming language, **[PyTorch](https://pytorch.org/)** library, **[OpenCV](https://opencv.org/)** and **[YOLOv5](https://github.com/ultralytics/yolov5)**

## Dataset

The total number of photos is 1008, of which 700 were used for training, 260 for validation, and 48 for testing. The number of images was increased by 3 times using the augmentation procedure, which was performed using the following filters: Flip Horizontal, Flip Vertical, Gaussian Blur 1.25px. All images were resized to 640×480 pixels.

> The dataset does not include banknotes of **UAH 1, UAH 2, UAH 5, and UAH 10**, so they were not used in the training.

## Training results

For training, we used the trained YOLOv5 Small model, which was further trained on the dataset described above. The following hardware was used to train the neural network: Xeon® E5-2683 v3 processor, 128 GB of RAM, using an RTX 3090 graphics processor (44 TFLOPS). The high-level programming language used for programming was Python 3.10.5 with the use of PyTorch 1.12 [23] and OpenCV 4.5.5 frameworks. Training time: about 35 minutes.

![cls_loss](https://i.imgur.com/ADD5YxH.png)

## Recognition results
![recognition results](https://i.imgur.com/fveDA9p.png)
