import sys
import os
import argparse
import cv2
import numpy as np
import scipy.misc
from models import baseline_nvidia_model
from keras.models import load_model

parser = argparse.ArgumentParser()

parser.add_argument("-train", action='store_true')
parser.add_argument("-test", action='store_true')
parser.add_argument("-test_and_vis", action='store_true')

parser.add_argument("-use_training_data", action='store_true')

train_images_opt_folder = 'data/training_images_opt/'
train_images_rgb_folder = 'data/training_images_rgb/'

# set up some constants for easy file i/o
test_images_opt_folder = 'data/test_images_opt/'
test_images_rgb_folder = 'data/test_images_rgb/'

BATCH_SIZE = 32
EPOCHS = 1

debug = False

if not debug:
    EPOCHS = 100

# i wanted to the files arranged numerically, which os.listdir doesn't inherently do.
def sort_files_numerically(path_to_files):
    files = os.listdir(path_to_files)
    for file in files:
        if(file.split(".")[1] != "jpg"):
            files.remove(file)

    return sorted(files, key=lambda x: int(x.split(".")[0]))

# calculate the optical flow between two frames and return a BGR image.
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

# load up the video and actually save the frames/optical flow output locally for faster training/testing.
def load_video(training=True):
    video = None
    if not training:
        video = cv2.VideoCapture('data/test.mp4')
    else:
        video = cv2.VideoCapture('data/train.mp4')

    success, first = video.read()

    fps = int(video.get(cv2.CAP_PROP_FPS))
    total_frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    print('Loading video %d seconds long with FPS %d and total frame count %d ' % (total_frame_count/fps, fps, total_frame_count))

    count = 0
    while success:
        success, second = video.read()
        if not success:
            break
        # go get the optical flow and save both the rgb image and optical flow image locally.
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

# simple function to handle all the image preprocessing stuff in one fucntion.
def process_image(file_name, training):
    if training:
        image = scipy.misc.imread(train_images_opt_folder + file_name)[200:400]
    else:
        image = scipy.misc.imread(test_images_opt_folder + file_name)[200:400]

    image = scipy.misc.imresize(image, [66, 200]) / 255
    if debug: scipy.misc.imsave('data/debug.jpg', image)
    return image

# go load data (images and speeds) for either the training set or the test set.
def get_data(training):
    if not training:
        image_file_names = sort_files_numerically(test_images_opt_folder)
        speed_data = np.loadtxt('data/train.txt')
    else:
        image_file_names = sort_files_numerically(train_images_opt_folder)
        speed_data = np.loadtxt('data/test.txt')

    images = []
    speeds = []

    count = 0
    for i in range(0, len(image_file_names) - 1):
        file_name = image_file_names[i]
        sys.stdout.write("\rProcessing %s" % file_name)

        # load optical flow image and also calculate the average speed between frame and frame + 1
        images.append(process_image(file_name, training))
        speeds.append((speed_data[i] + speed_data[i+1]) / 2)

        if debug and count == 1000:
            break
        count += 1

    print('\n')

    return np.asarray(images), np.asarray(speeds)


def train(training):
    X, y = get_data(training)
    model = baseline_nvidia_model(X.shape[1], X.shape[2], X.shape[3])
    model.fit(X, y, batch_size=BATCH_SIZE, epochs=EPOCHS, verbose=1, validation_split=0.2, shuffle=True)
    model.save('model.h5')

def test_and_visualize(training):
    X, y = get_data(training)
    model = load_model('model.h5')

    if not training:
        opt_file_names = sort_files_numerically(test_images_opt_folder)
        rgb_file_names = sort_files_numerically(test_images_rgb_folder)
    else:
        opt_file_names = sort_files_numerically(train_images_opt_folder)
        rgb_file_names = sort_files_numerically(train_images_rgb_folder)

    index = 0

    # credit: https://stackoverflow.com/questions/16615662/how-to-write-text-on-a-image-in-windows-using-python-opencv2/34273603
    font                   = cv2.FONT_HERSHEY_SIMPLEX
    fontScale              = 0.9
    fontColor              = (255,255,255)
    lineType               = 2

    for image_for_model, speed, opt_file, rgb_file in zip(X, y, opt_file_names, rgb_file_names):
        if not training:
            full_opt = scipy.misc.imread(test_images_opt_folder + opt_file)[200:400]
            full_rgb = scipy.misc.imread(test_images_rgb_folder + rgb_file)[200:400]
        else:
            full_opt = scipy.misc.imread(train_images_opt_folder + opt_file)[200:400]
            full_rgb = scipy.misc.imread(train_images_rgb_folder + rgb_file)[200:400]

        predicted_speed = model.predict(np.expand_dims(image_for_model, axis=0))[0][0]


        cv2.putText(full_rgb, "Actual: " + str(round(speed, 2)),
            (10,30),
            font,
            fontScale,
            fontColor,
            lineType)
        cv2.putText(full_rgb, "Predicted: " + str(round(predicted_speed,2)),
            (10,60),
            font,
            fontScale,
            fontColor,
            lineType)
        cv2.putText(full_rgb, "Error: " + str(round(abs(predicted_speed - speed), 2)),
            (10,90),
            font,
            fontScale,
            fontColor,
            lineType)

        cv2.imshow('frame', np.concatenate((full_opt, cv2.cvtColor(full_rgb, cv2.COLOR_BGR2RGB)), axis=0))

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
        index += 1

def evaluate_model(training):
    X, y = get_data(training)
    print("Loading/ Evaluating model... ")
    model = load_model('model.h5')
    loss_and_metrics = model.evaluate(X, y, batch_size=32)

    print("Done evaluating, found MSE to be...")
    print(loss_and_metrics)

def get_test_txt(training):
    X, y = get_data(training)
    print("Loading/ Evaluating model... ")
    text_file = open("test.txt", "w")
    model = load_model('model.h5')

    for image_for_model, speed in zip(X,y):
        pred = model.predict(np.expand_dims(image_for_model, axis=0))[0][0]
        text_file.write("%s\n" % pred)

    text_file.close()


if __name__ == '__main__':
    args = parser.parse_args()
    use_training_data = args.use_training_data

    if use_training_data:
        if not os.path.isdir(train_images_opt_folder) and not os.path.isdir(train_images_rgb_folder):
            os.mkdir(train_images_opt_folder)
            os.mkdir(train_images_rgb_folder)
            load_video(use_training_data)
    else:
        if not os.path.isdir(test_images_opt_folder) and not os.path.isdir(test_images_rgb_folder):
            os.mkdir(test_images_opt_folder)
            os.mkdir(test_images_rgb_folder)
            load_video(use_training_data)

    # get_test_txt(False)
    # sys.exit()

    if args.train:
        print("You chose to train a new model...")

        if not os.path.exists('data/train.txt'):
            print("Can't find train.txt")
            sys.exit()

        if not os.path.exists('data/train.mp4'):
            print("Can't find train.mp4")
            sys.exit()

        train(use_training_data)

    elif args.test:
        print("You chose to test the model...")
        evaluate_model(use_training_data)

    elif args.test_and_vis:
        print("You chose to test and visualize the output of the model...")
        test_and_visualize(use_training_data)
