from keras import losses
from keras.models import Model, Sequential
from keras.layers import Conv2D, Dense, Activation, Flatten, LSTM, BatchNormalization, TimeDistributed, Dropout, Convolution2D, Conv3D, MaxPooling3D
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

def get_model_3D_NVIDIA(samples, height, width, channels):
    model = Sequential()

    model.add(Conv3D(24, (3), strides=(1), input_shape=(samples, height, width, channels), activation='relu', padding='same'))
    model.add(MaxPooling3D(pool_size=(1,2,2), strides=(1,2,2), padding='valid'))
    model.add(Conv3D(36, (3), strides=(1), activation='relu', padding='same'))
    model.add(MaxPooling3D(pool_size=(2,2,2), strides=(2,2,2), padding='valid'))
    model.add(Conv3D(48, (3), strides=(1), activation='relu', padding='same'))
    model.add(MaxPooling3D(pool_size=(2,2,2), strides=(2,2,2), padding='valid'))
    model.add(Conv3D(64, (3), strides=(1), activation='relu', padding='same'))

    model.add(Flatten())
    model.add(Dropout(0.8))
    
    model.add(Dense(100))
    model.add(Activation('relu'))
    model.add(Dropout(0.7))

    model.add(Dense(50))
    model.add(Activation('relu'))
    model.add(Dropout(0.7))

    model.add(Dense(10))
    model.add(Activation('relu'))

    model.add(Dense(1))
    model.add(Activation('linear'))

    model.compile(optimizer=optimizers.Adam(lr=0.0001), loss=losses.mean_squared_error)
    model.summary()

    return model

def dense_net():
    model = DenseNet121(include_top=False, input_shape=(224,224,3))

    # for layer in model.layers:
    #     if layer.name.startswith('batch_normalization_'):
    #
    x = Flatten()(model.output)
    x = Dense(128, activation = 'linear')(x)
    x = Dropout(0.7)(x)
    x = Dense(100, activation = 'linear')(x)
    x = Dropout(0.5)(x)
    x = Dense(50, activation = 'linear')(x)
    x = Dropout(0.5)(x)
    x = Dense(10, activation = 'linear')(x)
    predictions = Dense(1, activation = 'linear')(x)
    head_model = Model(input = model.input, output = predictions)
    head_model.compile(optimizer=optimizers.Adam(lr=0.0001), loss=losses.mean_squared_error)
    head_model.summary()

    return head_model

get_model_3D_NVIDIA(4, 66,200,3)
