# Lil-Nvidia

This net is called Lil Nvidia because I like rappers who have a "Lil" in front of their name (ex. Lil Peep) and because its a smaller version of the NVIDIA net used for the task of learning to steer a vehicle in and end to end manner.

Lil Nvidia specializes in predicting the speed of a vehicle from just a video.

To *test* this and simply output an MSE value over some test video, first create a folder called ```data``` in the root. In this folder, include your own ```test.mp4``` and ```test.txt```. They must be named just like this. Then simply run:
```sh
python all.py -test
```

To *train* this, add your own ```train.mp4``` and ```train.txt``` to the ```data```. They must be named just like this. Then simply run:
```sh
python all.py -train -use_training_data
```


#Dependencies:
- OpenCV 3.4.0 pip install opencv-python
- NumPy 1.14.0 pip install numpy
- SciPy 1.0.0 pip install scipy
- TensorFlow 1.15.0 pip install tensorflow
- Keras 2.1.3 pip install keras

Oh and you'll also need Pillow pip install pillow
