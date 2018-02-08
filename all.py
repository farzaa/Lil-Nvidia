import sys
import cv2
import os
import numpy as np
import scipy.misc
from models import baseline_nvidia_model, get_model_3D_NVIDIA, nvidia_lstm

from keras import losses
from keras.models import Model, Sequential
from keras.layers import Conv2D, Dense, Activation, Flatten, LSTM, BatchNormalization, TimeDistributed, Dropout, Convolution2D
from keras import optimizers

from keras.models import load_model
import argparse


import random
import time

parser = argparse.ArgumentParser()

parser.add_argument("-train", action='store_true')
parser.add_argument("-test", action='store_true')
parser.add_argument("-test_and_vis", action='store_true')

parser.add_argument("-input_video")
parser.add_argument("-input_txt", default='train.txt')

BATCH_SIZE = 32
EPOCHS = 1

debug = True
train_images_opt_folder = 'data/training_images_opt/'
train_images_rgb_folder = 'data/training_images_rgb/'

test_images_opt_folder = 'data/test_images_opt/'
test_images_rgb_folder = 'data/test_images_rgb/'

if not debug:
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
    hsv[...,1] = 255

    first = cv2.cvtColor(first, cv2.COLOR_BGR2GRAY)
    second = cv2.cvtColor(second, cv2.COLOR_BGR2GRAY)

    # https://funvision.blogspot.cz/2016/02/opencv-31-tutorial-optical-flow.html
    flow = cv2.calcOpticalFlowFarneback(first, second, None, 0.4, 1, 12, 2, 8, 1.2, 0)
    mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
    hsv[...,0] = ang*180/np.pi/2
    hsv[...,2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    return bgr

def load_video(training=True):
    video = cv2.VideoCapture('data/train.mp4')
    if not training:
        video = cv2.VideoCapture('data/test.mp4')

    success, first = video.read()

    fps = int(video.get(cv2.CAP_PROP_FPS))
    total_frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    print('Loading video %d seconds long with FPS %d and total frame count %d ' % (total_frame_count/fps, fps, total_frame_count))

    count = 0
    while success:
        #TODO
        success, second = video.read()
        if not success:
            break

        flow = get_dense_opt_flow(first, second)
        if training:
            cv2.imwrite(train_images_opt_folder + str(count) + '.jpg', flow)
            cv2.imwrite(train_images_rgb_folder + str(count) + '.jpg', second)
        else:
            cv2.imwrite(test_images_opt_folder + str(count) + '.jpg', flow)
            cv2.imwrite(test_images_rgb_folder + str(count) + '.jpg', second)

        first = second
        sys.stdout.write("\rCurrently on frame %d of video. Processing with optical flow." % count)
        count += 1

    print('Saved %d frames' % (count) )
    video.release()


def process_image(file_name, training):
    image = scipy.misc.imread('data/training_images_opt/' + file_name)[200:400]
    if not training:
        image = scipy.misc.imread('data/test_images_opt/' + file_name)[200:400]
    image = scipy.misc.imresize(image, [66, 200]) / 255
    if debug: scipy.misc.imsave('data/debug.jpg', image)
    return image

def get_data(training=True):
    image_file_names = sort_files_numerically(train_images_opt_folder)
    speed_data = np.loadtxt('data/train.txt')
    if not training:
        image_file_names = sort_files_numerically(test_images_opt_folder)
        #TODO
        speed_data = np.loadtxt('data/train.txt')

    images = []
    speeds = []
    rgb = []
    full_opt = []
    for i in range(0, len(image_file_names) - 1):
        file_name = image_file_names[i]
        sys.stdout.write("\rProcessing %s" % file_name)

        images.append(process_image(file_name, training))
        speeds.append((speed_data[i] + speed_data[i+1]) / 2)

        if debug and i == 21000:
            break

    print('\n')

    return np.asarray(images), np.asarray(speeds)


def train(speeds_file_name):
    X, y = get_data(training=True)
    model = baseline_nvidia_model(X.shape[1], X.shape[2], X.shape[3])
    model.fit(X, y, batch_size=BATCH_SIZE, epochs=EPOCHS, verbose=1, validation_split=0.2, shuffle=True)
    model.save('comma_model.h5')


def test_and_visualize():
    X, y = get_data(training=False)
    model = load_model('comma_model.h5')

    opt_file_names = sort_files_numerically(test_images_opt_folder)
    rgb_file_names = sort_files_numerically(train_images_opt_folder)

    index = 0
    for image_for_model, speed, opt_file, rgb_file in zip(X, y, opt_file_names, rgb_file_names):
        full_opt = scipy.misc.imread(test_images_opt_folder + opt_file)[200:400]
        full_rgb = scipy.misc.imread(test_images_rgb_folder + rgb_file)[200:400]

        cv2.imshow('frame', np.concatenate((full_opt, full_rgb), axis=0))
        print(speed)
        print(model.predict(np.expand_dims(image_for_model, axis=0)))

        if cv2.waitKey(100) & 0xFF == ord('q'):
            break
        index += 1

def evaluate_model():
    X, y = get_data(training=True)
    print("Loading/ Evaluating model... ")
    model = load_model('comma_model.h5')
    loss_and_metrics = model.evaluate(X, y, batch_size=32)
    print(loss_and_metrics)


if __name__ == '__main__':
    args = parser.parse_args()

    input_video = args.input_video
    input_txt = args.input_txt

    if not os.path.exists('data/' + input_video):
        print("Can't find input video...")
        sys.exit()

    print("Found input video %s" % input_video)

    if args.train:

        print("You chose the training option...")
        # input_txt = args.input_txt
        # if not os.path.exists('data/' + input_txt):
        #     print("Can't find input txt ground truth...")
        #     sys.exit()
        #
        # if not os.path.isdir('data/training_images_opt'):
        #     os.mkdir('data/training_images_opt')
        # if not os.path.isdir('data/training_images_opt'):
        #     os.mkdir('data/training_images_rgb')
        #
        #
        # load_video(train_video_name)
        # build_data_locally(input_video)
        # train(input_txt)

    elif args.test:
        if not os.path.isdir(test_images_opt_folder) and not os.path.isdir(test_images_rgb_folder):
            os.mkdir(test_images_opt_folder)
            os.mkdir(test_images_rgb_folder)
            load_video(training=False)
        evaluate_model()

    elif args.test_and_vis:
        print("Testing and visualizing...")
        if not os.path.isdir(test_images_opt_folder) and not os.path.isdir(test_images_rgb_folder):
            os.mkdir(test_images_opt_folder)
            os.mkdir(test_images_rgb_folder)
            load_video(training=False)

        test_and_visualize()

    elif args.just_visualize:
        print("Just visualizing...")


    args = parser.parse_args()
    print(args)
