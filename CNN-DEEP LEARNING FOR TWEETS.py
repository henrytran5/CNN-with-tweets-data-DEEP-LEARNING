
# CONVOLUTIONAL NEURAL NETWORK (CNN)
from wordcloud import WordCloud,STOPWORDS
from nltk.corpus import stopwords
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize

# Import dataset from csv, tsv or txt files
import pandas as pd
# Import library for mathematical computations, numerical works
import numpy as np

# LIBRARIES FOR LAYERS OF CNN METHOD

from keras.models import Sequential

from keras.layers import Dense, Dropout, Activation

from keras.layers import Embedding

from keras.layers import Conv1D, MaxPooling1D

# Import LSTM layer

from keras.layers import LSTM

# Call function of plot
import matplotlib.pyplot as plt

# Set up for Embedding

max_features = 10000 #input length of sentence

maxlen = 38854 # input size, word size

embedding_size = 128 # output size, vector space

# Convolution

kernel_size = 5 #the length of the 1D convolution.

filters = 64 #the number of output filters in the convolution by integer

pool_size = 4
# LSTM

lstm_output_size = 70

# Training

batch_size = 30

# Epoch time in running

epochs = 3
# Importing the dataset

#TWEETER DATASETS

#dataset = pd.read_csv('tweeter-dev-full-A.csv')
#dataset = pd.read_csv('tweeter-dev-full-B.csv')

#HAPPY AND SURPRISE DATASETS 
dataset = pd.read_csv('tweet2.csv')
dt = dataset.iloc[:,[1,2]]
# Emotion Datasets

# Large dataset
#dataset = pd.read_csv('master40k_textblob.csv')
#dataset = pd.read_csv('master_dataset_40k.csv')
#dataset = pd.read_csv('textemotion2.csv')
# Small dataset
#dataset = pd.read_csv('EmotionPhrases.csv')
#dataset = pd.read_csv('E:\EmotionWords.csv')

# Cleaning the texts
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

corpus = []
for i in range(0,len(dt['A'])): # length of dataset
    review = re.sub('[^a-zA-Z]', ' ', dt['A'][i])
    review = review.lower()
    review = review.split()
    ps = PorterStemmer()
    stopword_set = set(stopwords.words('english'))
    review = [ps.stem(word) for word in review if not word in stopword_set]
    review = ' '.join(review)
    corpus.append(review)

# Creating the Bag of Words model
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features)
X = cv.fit_transform(corpus).toarray()
y = dt.iloc[:, 1].values


# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=0)

print(len(X_train), 'train sequences')

print(len(X_test), 'test sequences')


print('Build model...')

# Models for CNN

model = Sequential()

model.add(Embedding(max_features, embedding_size, input_length=maxlen))

model.add(Dropout(0.25))

# Model of Conv1D, MaxPooling1D

model.add(Conv1D(filters,

                 kernel_size,

                 padding='valid',

                 activation='relu',

                 strides=1))

model.add(MaxPooling1D(pool_size=pool_size))

# Model of LSTM

model.add(LSTM(lstm_output_size))

model.add(Dense(1))

model.add(Activation('sigmoid'))

model.compile(loss='binary_crossentropy',

              optimizer='adam',

              metrics=['accuracy'])

print('Train...')

history=model.fit(X_train, y_train,

          batch_size=batch_size,

          epochs=epochs,

          validation_data=(X_test, y_test))

# Get training and test loss histories

score, acc = model.evaluate(X_test, y_test, batch_size=batch_size)

print('Test score and Test accuracy:')

print('Test score:', score)

print('Test accuracy:', acc)

print(history.history.keys())

# Final evaluation of the model


# Get training and test loss histories
training_loss = history.history['loss']
test_loss = history.history['val_loss']

# Create count of the number of epochs
epoch_count = range(1, len(training_loss) + 1)

# Visualize loss history
plt.plot(epoch_count, training_loss, 'gd--')
plt.plot(epoch_count, test_loss, 'bo--')
plt.legend(['Training Loss', 'Test Loss'])
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Model Loss')
plt.show()

# Get training and test accuracy histories
training_accuracy = history.history['acc']
test_accuracy = history.history['val_acc']

# Visualize Accuracy history
plt.plot(epoch_count, training_accuracy , 'ko--')
plt.plot(epoch_count, test_accuracy , 'rd--')
plt.legend(['Training accuracy ', 'Test accuracy '], loc='upper left')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Model Accuracy')
plt.show()
