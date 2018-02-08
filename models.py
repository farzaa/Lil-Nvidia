from keras.models import Model, Sequential
from keras.layers import Conv2D, Dense, Activation, Flatten, BatchNormalization, Dropout
from keras import optimizers

def baseline_nvidia_model(height, width, channels):
    model = Sequential()

    model.add((BatchNormalization(epsilon=0.001, axis=1, input_shape=(height, width, channels))))

    model.add(Conv2D(24, (5,5), strides=(2,2), activation='relu'))
    model.add(Conv2D(36, (5,5), strides=(2,2), activation='relu'))
    model.add(Conv2D(48, (5,5), strides=(2,2), activation='relu'))
    model.add(Conv2D(64, (3,3), strides=(1,1), activation='relu'))


    model.add(Flatten())
    model.add(Dropout(0.8))

    model.add(Dense(100, activation='relu'))
    model.add(Dropout(0.7))

    model.add(Dense(50, activation='relu'))
    model.add(Dropout(0.7))

    model.add(Dense(10, activation='relu'))
    model.add(Dense(1, activation='linear'))

    model.compile(optimizer=optimizers.Adam(lr=0.0001), loss='mse')
    model.summary()
    return model
