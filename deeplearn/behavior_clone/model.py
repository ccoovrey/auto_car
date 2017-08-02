# load data from csv
import csv
import cv2
import numpy as np
from sklearn.model_selection import train_test_split

samples = []
with open('../../../data/bc_many/driving_log.csv') as csvfile:
    rd = csv.reader(csvfile)
    for line in rd:
        samples.append(line)
        
train_samples, validation_samples = train_test_split(samples, test_size=0.2)

#import sklearn
from sklearn.utils import shuffle

# path to training data
curr_pth = '../../../data/bc_many/IMG/'

def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        shuffle(samples)
        
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]
            images = []
            measurements = []
            
            for batch_sample in batch_samples:
                # images
                img_center = cv2.imread(curr_pth + batch_sample[0].split('/')[-1])
                img_left = cv2.imread(curr_pth + batch_sample[1].split('/')[-1])
                img_right = cv2.imread(curr_pth + batch_sample[2].split('/')[-1])
                images.extend([img_center, img_left, img_right, cv2.flip(img_center, 1), cv2.flip(img_left, 1),
                               cv2.flip(img_right, 1)])
                
                # measurements
                steer_center = float(batch_sample[3])
                correction = 0.2 # parameter to tune
                steer_left = steer_center + correction
                steer_right = steer_center - correction
                measurements.extend([steer_center, steer_left, steer_right, steer_center*-1.0, steer_left*-1.0,
                                     steer_right*-1.0])

            # trim image to only see section with road
            X_train = np.array(images)
            y_train = np.array(measurements)
            #yild sklearn.utils.shuffle(X_train, y_train)
            yield shuffle(X_train, y_train)
   
# model architecture
from keras.models import Sequential
from keras.layers import Flatten, Dense, Convolution2D, Dropout, MaxPooling2D, Lambda, Cropping2D
input_shape=(160,320,3)

model = Sequential()
model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(160,320,3)))
model.add(Cropping2D(cropping=((70,25), (0,0))))
model.add(Convolution2D(24,5,5,subsample=(2,2),activation='relu'))
model.add(Convolution2D(36,5,5,subsample=(2,2),activation='relu'))
model.add(Convolution2D(48,5,5,subsample=(2,2),activation='relu'))
model.add(Convolution2D(64,3,3,activation='relu'))
model.add(Convolution2D(64,3,3,activation='relu'))
model.add(Dropout(0.5))
model.add(Flatten())
model.add(Dense(250,activation='relu'))
model.add(Dense(1))
model.summary()

# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=32)
validation_generator = generator(validation_samples, batch_size=32)

model.compile(loss='mse', optimizer='adam')
model.fit_generator(train_generator, samples_per_epoch= \
                    len(train_samples), validation_data=validation_generator, \
                    nb_val_samples=len(validation_samples), nb_epoch=3)

# save model
model.save('model.h5')