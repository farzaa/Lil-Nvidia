# Lil-Nvidia

This net is called Lil Nvidia because I like rappers who have a "Lil" in front of their name (ex. Lil Peep) and because its inspired by the NVIDIA net used originally for the task of learning to steer a vehicle using a single end to end ConvNet.

![GIFHERE](https://github.com/farzaa/Lil-Nvidia/blob/master/demo.gif?raw=true)

Lil Nvidia specializes in predicting the speed of a vehicle from just a video. Please take a look at my section titled "Logic" for more info about how this thing works in the background.

### Basics
Note: This script first breaks the video into individual frames, calculates optical flow, and saves both the RGB data and optical flow data *locally*. Then, this data is all loaded into memory at one time when its time to train or test. This may be an issue depending on how much RAM your machine has. Future work would include to have this train by batch.

To *test* this and simply output an MSE value over some test video, first create a folder called ```data``` in the root. In this folder, include your own ```test.mp4``` and ```test.txt```. They must be named just like this. Then simply run:
```sh
python all.py -test
```

To *train* this, add your own ```train.mp4``` and ```train.txt``` to the ```data``` folder. They must be named just like this. Then simply run:
```sh
python all.py -train -use_training_data
```


### Dependencies:
- OpenCV 3.4.0 ```pip install opencv-python```
- NumPy 1.14.0 ```pip install numpy```
- SciPy 1.0.0 ```pip install scipy```
- TensorFlow 1.15.0 ```pip install tensorflow```
- Keras 2.1.3 ```pip install keras```

Oh and you'll also need Pillow ```pip install pillow```.


### Logic:
This was all done for a 7 day challenge, so I had to think of approaches that would work well and fast because I didn't have much GPU compute. Here are things I tried before settling on the current iteration. For all these approaches, speed was the output.

- 2D CNN with a *single* frame as input. Didn't learn.
- 2D CNN with a frame *stacked* with the next frame from the video. Didn't learn.
- 3D CNN with 4 frames sent together. This approach was actually giving very good early results but was taking FOREVER to train. Had to bail from it.
- 2D CNN with a optical flow data AND RGB data sent together. Didn't learn.
- 2D CNN where the optical flow data and RGB data were blended together using ```cv2.addWeighted```. This was a crazy idea that sorta worked, but the net stopped learning after a certain point.
- 2D CNN with just optical flow passed in as BGR image. So, this gave very good results and I didn't expect it to. But, it overfit terribly.

The last option is what I decided to iterate upon since it was the only version doing very well on the training set but terribly on the actual validation set. Originally, I was using the full NVIDIA CNN which has about 1.5 million parameters. I figured this was to much! I messed with the NVIDIA architecture about 20 different times and finally found a good balance of dropout, dense layer parameters, and conv layer parameters. Without dropout, this thing overfits horribly which is why in the first dense layer I call have a very high dropout parameter of 0.8.

If you have anty questions, please drop me a DM on Twitter @farzatv. And enjoy this picture of Lil Peep :)
![LilPeep](https://i.imgur.com/pIk0rTO.jpg)
