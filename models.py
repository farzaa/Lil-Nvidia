from keras import losses
from keras.models import Model, Sequential
from keras.layers import Conv2D, Dense, Activation, Flatten, LSTM, BatchNormalization, TimeDistributed, Dropout, Convolution2D
from keras import optimizers
from keras.applications.densenet import DenseNet121

def baseline_nvidia_model(height, width, channels):
    model = Sequential()
    model.add(BatchNormalization(epsilon=0.001, axis=1, input_shape=(height, width, channels)))
    model.add(Conv2D(24, (5,5), strides=(2,2), input_shape=(height, width, channels)))
    model.add((Activation('relu')))
    model.add(Conv2D(36, (5,5), strides=(2,2)))
    model.add(Activation('relu'))
    model.add(Conv2D(48, (5,5), strides=(2,2)))
    model.add(Activation('relu'))
    model.add(Conv2D(64, (3,3), strides=(1,1)))
    model.add(Activation('relu'))
    model.add(Conv2D(64, (3,3), strides=(1,1)))
    model.add(Activation('relu'))

    model.add(Flatten())
    model.add(Dropout(0.5))
    model.add(Dense(1164))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(100))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(50))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(10))
    model.add(Activation('relu'))

    model.add(Dense(1))
    model.add(Activation('linear'))

    model.compile(optimizer=optimizers.Adam(lr=0.0001), loss=losses.mean_squared_error)
    model.summary()

    return model

# def dense_net():
#     model = DenseNet121()
#     model.summary(include_top=False, layers.Input(input_shape=(66, 200, 3)))
#     print("done")
#
# # def two_stream_model(height, width, channels):
# #     first = Input(shape=(height, width, channels))
# #     a = Conv2D(24, (5,5), strides=(2,2))(first)
# #     a = Flatten()(a)
# #
# #     second = Input(shape=(height,width,channels))
# #     b = Conv2D(24, (5,5), strides=(2,2))(second)
# #     b = Flatten()(b)
# #
# #     model = Model(inputs = main_input, outputs = [pred_out, deconv_out])
#
# dense_net()
