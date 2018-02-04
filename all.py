import sys
import cv2
import os
import numpy as np
import scipy.misc
from models import baseline_nvidia_model

from keras import losses
from keras.models import Model, Sequential
from keras.layers import Conv2D, Dense, Activation, Flatten, LSTM, BatchNormalization, TimeDistributed, Dropout, Convolution2D
from keras import optimizers

BATCH_SIZE = 32
EPOCHS = 1
debug = True

if !debug:
    EPOCHS = 100

def sort_files_numerically(path_to_files):
    files = os.listdir(path_to_files)

    for file in files:
        if(file.split(".")[1] != "jpg"):
            files.remove(file)

    return sorted(files, key=lambda x: int(x.split(".")[0]))

def get_dense_opt_flow(first, second):
    hsv = np.zeros_like(first)

    # don't care about saturation.
    hsv[...,1] = 0

    first = cv2.cvtColor(first, cv2.COLOR_BGR2GRAY)
    second = cv2.cvtColor(second, cv2.COLOR_BGR2GRAY)

    # https://funvision.blogspot.cz/2016/02/opencv-31-tutorial-optical-flow.html
    flow = cv2.calcOpticalFlowFarneback(first, second, None, 0.4, 1, 12, 2, 8, 1.2, 0)
    mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
    hsv[...,0] = ang*180/np.pi/2
    hsv[...,2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    return bgr

def build_data_locally(video_name):
    # check if data is already available locally.
    if not os.path.isdir('data/training_images_rgb'):
        os.mkdir('data/training_images_rgb')

    if not os.path.isdir('data/training_images_opt'):
        os.mkdir('data/training_images_opt')

    else:
        print("Data already found in filesystem")
        return


    video = cv2.VideoCapture('data/' + video_name)
    success, first = video.read()

    fps = int(video.get(cv2.CAP_PROP_FPS))
    total_frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    print('Loading video %d seconds long with FPS %d and total frame count %d ' % (total_frame_count/fps, fps, total_frame_count))

    count = 0
    while success:

        cv2.imwrite('data/training_images_rgb/' + str(count) + '.jpg', first)
        success, second = video.read()

        flow = get_dense_opt_flow(first, second)
        cv2.imwrite('data/training_images_opt/' + str(count) + '.jpg', flow)

        first = second
        sys.stdout.write("\rCurrently on frame %d of video. Processing with optical flow." % count)
        count += 1

    print('Saved %d frames' % (count) )
    video.release()

def process_image(file_name):
    image = scipy.misc.imread('data/training_images_opt/' + file_name)[200:400]
    rgb_image = scipy.misc.imread('data/training_images_rgb/' + file_name)[200:400]

    combined_image = cv2.addWeighted(image,1.0, rgb_image,1.0,0)
    combined_image = scipy.misc.imresize(combined_image, [66, 200]) / 255
    if debug: scipy.misc.imsave('data/debug.jpg', combined_image)
    return combined_imageq

def get_training_data():
        image_file_names = sort_files_numerically('data/training_images_opt')
        speed_data = np.loadtxt('data/train.txt')

        images = []
        idea_images = []
        speeds = []
        for i in range(len(image_file_names) - 1):
            file_name = image_file_names[i]
            sys.stdout.write("\rProcessing %s" % file_name)

            images.append(process_image(file_name))
            speeds.append((speed_data[i] + speed_data[i + 1]) / 2)

            if debug and i == 70:
                break

        print('\n')
        return np.asarray(images), np.asarray(speeds)

def train(X, y):
    model = baseline_nvidia_model(X.shape[1], X.shape[2], X.shape[3])
    model.fit(X, y, batch_size=BATCH_SIZE, epochs=EPOCHS, verbose=1, validation_split=0.2, shuffle=True)
    model.save('comma_model.h5')

def debug_visualize(X, y):
    for image, speed in zip(X,y):
        cv2.imshow('frame', image)
        print(speed)
        if cv2.waitKey(50) & 0xFF == ord('q'):
            break



if __name__ == '__main__':
    build_data_locally('train.mp4')
    X, y = get_training_data()
    train(X, y)
