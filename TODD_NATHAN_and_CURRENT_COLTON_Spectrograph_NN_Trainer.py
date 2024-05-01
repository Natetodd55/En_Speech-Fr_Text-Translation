import os
import pathlib
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import librosa
from tensorflow.keras import layers
from tensorflow.keras import models
#from IPython import display
#import sounddevice as sd
from scipy.io.wavfile import write
from datasets import load_dataset
from silence_tensorflow import silence_tensorflow
silence_tensorflow()

###################################################
#Functions needed
###################################################


# Convert raw data to a SPECTROGRAM, a 2D image that represents the frequency info
def get_spectrogram(waveform):
    # Convert the waveform to a spectrogram via a STFT.
    spectrogram = tf.signal.stft(waveform, frame_length=255, frame_step=128)
    # Obtain the magnitude of the STFT.
    spectrogram = tf.abs(spectrogram)
    #shifting the data to a tensor format
    spectrogram = spectrogram[..., tf.newaxis]
    #returning the data as a np array
    return np.asarray(spectrogram).astype(np.float32)


def preprocess_data(dataset):
    # Preprocess the dataset: Extract Spectrogram and labels
    #creating empty lists to hold the data
    x = []
    y = []
    #creating a dictionary to keep track of the sample count
    data_dict = {}
    #looping through our data and seperating the audio file and the label
    for i, item in enumerate(dataset):
        audio, label = item["audio"]["array"], item["label"]
        #checking to see if the label was already in our dictionary and putting it in if not.
        if label not in data_dict:
            data_dict[label] = 1
        else:
            #tracking the number of samples we are using for training
            if data_dict[label] <= 1500:
                #calling the spectrogram function to convert our audio files.
                spectrogram = get_spectrogram(audio)
                #using samples of only one shape size because the dataset has different sizes
                if spectrogram.shape == (124, 129, 1):
                    #appending the data to the lists and updating the sample counts in the dictionary
                    x.append(spectrogram)
                    y.append(label)
                    data_dict[label] += 1
    return np.asarray(x), np.asarray(y)


###################################################
#End of Functions needed
###################################################

#loading the dataset from hugging face using the dataset library
print("Loading Dataset")
dataset = load_dataset("speech_commands", "v0.02")

#sending the training section of the data to the preprocess function
print("Processing Dataset")
train_data, train_labels = preprocess_data(dataset['train'])

#grabbing our input shape from the first entry of the preprocessed data, we took measures to ensure all sample are the same size
input_shape = train_data[0].shape
#building our model
model = models.Sequential([
    #creating our input layer
    layers.Input(shape=input_shape),
    #downsizing our data to lower training time
    layers.Resizing(32, 32),
    #using convolution layers for image recognition on our spectrogram data
    layers.Conv2D(32, 3, activation='relu'),
    layers.Conv2D(64, 3, activation='relu'),
    layers.MaxPooling2D(),
    #dropping out unneeded layers
    layers.Dropout(0.25),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    #moving down on layers to just the amount of class labels
    layers.Dense(len(train_labels)),
])

#summarizing and compiling our model
model.summary()
model.compile(optimizer=tf.keras.optimizers.Adam(), loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])

#training our model
EPOCHS = 15
history = model.fit(train_data, train_labels, epochs=EPOCHS)

#saving our model
model.save('command.h5', include_optimizer=True)
