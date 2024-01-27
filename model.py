from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Flatten, Dense, Dropout

input_shape = (66, 200, 3)

def buildModel():
    model = Sequential()

    # 5x5 Convolutional layers with stride of 2x2
    model.add(Conv2D(24, (5, 5), strides=(2, 2), activation='elu', input_shape=input_shape))
    model.add(Conv2D(36, (5, 5), strides=(2, 2), activation='elu'))
    model.add(Conv2D(48, (5, 5), strides=(2, 2), activation='elu'))

    # 3x3 Convolutional layers with stride of 1x1
    model.add(Conv2D(64, (3, 3), activation='elu'))
    model.add(Conv2D(64, (3, 3), activation='elu'))

    model.add(Flatten())
    # Three fully connected layers
    model.add(Dense(100, activation='elu'))
    model.add(Dropout(.25))
    model.add(Dense(50, activation='elu'))
    model.add(Dropout(.25))
    model.add(Dense(10, activation='elu'))
    model.add(Dropout(.25))

    # Output layer with linear activation
    model.add(Dense(1, activation="linear"))

    return model