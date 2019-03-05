# Task 2
# Importing library

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
% matplotlib inline
plt.rcParams["figure.dpi"] = 120
np.set_printoptions(precision=3, suppress=True)
import keras

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers.advanced_activations import LeakyReLU, PReLU
from keras.optimizers import SGD
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.grid_search import GridSearchCV
from keras.models import model_from_json

np.random.RandomState(seed=0)

# Part 1 using traditional train and test train model on mnist dataset
from keras.datasets import mnist

(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Here we have divided train test by 255 in order to bring them on 0-1 scale
X_train = X_train.reshape(60000, 784).astype('float32') / 255
X_test = X_test.reshape(10000, 784).astype('float32') / 255

print (X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)


def make_model(optimizer='adam', hidden_size=32):
    model = Sequential([
        Dense(hidden_size, input_shape=[784,]),
        Activation('relu'),
        Dense(hidden_size),
        Activation('tanh'),
        Dense(10),
        Activation('softmax')])
        
    model.compile(optimizer=optimizer, loss = 'categorical_crossentropy', metrics=['accuracy'])
    return model

clf = KerasClassifier(make_model)

num_classes=10
y_train= keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

# fitting Multi-layer perceptron to the data using keras
#model1=clf.fit(X_train, y_train, validation_split=0.2,epochs=20, verbose=1)
model1=clf.fit(X_train, y_train, validation_data=(X_test, y_test),epochs=20, verbose=1)

# Visulize the model1
# summarize history for accuracy
plt.plot(model1.history['acc'])
plt.plot(model1.history['val_acc'])
plt.title('model using traditional Train test split : learning curve')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# as we see no significant gain after 5 epoch we check the score on test dataset

model1 = make_model()
model1.fit(X_train, y_train,epochs=5)

# model evaluation  on test data
score = model1.evaluate(X_test, y_test, verbose=0)

print("\nTest loss: {:.3f}".format(score[0])) # Test loss: 0.115
print("Test Accuracy: {:.3f}".format(score[1])) # Test Accuracy: 0.966

# saving the model if you do not want to train model again which usually saves time

# serialize model to JSON
model_json = model1.to_json()
with open("Task2_model1.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model1.save_weights("Task2_model1.h5")
print("Saved model to disk")


# to load the saved model
'''
# load json and create model
json_file = open('Task2_model1.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("Task2_model1.h5")
print("Loaded model from disk")

# prediction using the stored model

# evaluate loaded model on test data
loaded_model.compile(optimizer='adam', loss = 'categorical_crossentropy', metrics=['accuracy'])
loaded_model_score = loaded_model.evaluate(X_test, y_test, verbose=0)
print("\nTest loss: {:.3f}".format(loaded_model_score[0])) # Test loss: 0.115
print("Test Accuracy: {:.3f}".format(loaded_model_score[1])) # Test Accuracy: 0.966
'''

'''

# grid search

param_grid = {'epochs' : [1, 3, 5]}

grid = GridSearchCV(clf, param_grid = param_grid, cv = 5)
grid.fit(X_train, y_train)

print(grid.best_params_)
# we have got epochs=5 as best 
# we confirm the same from the below graph as well
'''
# Part -2 : Using 10000 samples from train data for validation data set
# we need to use 10000 samples from train data so we are doing the same in below code

# re reading the data

np.random.RandomState(seed=0)
from keras.datasets import mnist

(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Here we have divided train test by 255 in order to bring them on 0-1 scale
X_train = X_train.reshape(60000, 784).astype('float32') / 255
X_test = X_test.reshape(10000, 784).astype('float32') / 255


def make_model(optimizer='adam', hidden_size=32):
    model = Sequential([
        Dense(hidden_size, input_shape=[784,]),
        Activation('relu'),
        Dense(hidden_size),
        Activation('tanh'),
        Dense(10),
        Activation('softmax')])
        
    model.compile(optimizer=optimizer, loss = 'categorical_crossentropy', metrics=['accuracy'])
    return model

clf = KerasClassifier(make_model)

# taking subset fro train data for validation 

X1=X_train[:50000]
Y1=y_train[:50000]
X2=X_train[50000:]
Y2=y_train[50000:]
num_classes=10
Y1 = keras.utils.to_categorical(Y1, num_classes)
Y2 = keras.utils.to_categorical(Y2, num_classes)

model2 =clf.fit(X1, Y1, validation_data=(X2, Y2),epochs=20, verbose=1)

#history =clf.fit(X_train, y_train,validation_split=0.33, epochs=50, verbose=1)

# list all data in history
print(model2.history.keys())
# summarize history for accuracy
plt.plot(model2.history['acc'])
plt.plot(model2.history['val_acc'])
plt.title('model using 10000 sample from train data as test dataset learning curve')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# lets evaluate on test data
# as we see no significant gain after 5 epoch we check the score on test dataset

model2 = make_model()
model2.fit(X1, Y1,epochs=5)

# model evaluation  on test data
num_classes=10
y_test = keras.utils.to_categorical(y_test, num_classes)

score = model2.evaluate(X_test, y_test, verbose=0)

print("\nTest loss: {:.3f}".format(score[0])) # Test loss: 0.123
print("Test Accuracy: {:.3f}".format(score[1])) # Test Accuracy: 0.962

# saving the model if you do not want to train model again which usually saves time

# serialize model to JSON
model_json = model2.to_json()
with open("Task2_model2.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model2.save_weights("Task2_model2.h5")
print("Saved model to disk")


# to load the saved model
'''
# load json and create model
json_file = open('Task2_model2.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("Task2_model2.h5")
print("Loaded model from disk")

# prediction using the stored model

# evaluate loaded model on test data
loaded_model.compile(optimizer='adam', loss = 'categorical_crossentropy', metrics=['accuracy'])
loaded_model_score = loaded_model.evaluate(X_test, y_test, verbose=0)
print("\nTest loss: {:.3f}".format(loaded_model_score[0])) # Test loss: 0.123
print("Test Accuracy: {:.3f}".format(loaded_model_score[1])) # Test Accuracy: 0.962
'''

# part 3
# Compare a “vanilla” model with a model using drop-out

# laoding the data again

np.random.RandomState(seed=0)
from keras.datasets import mnist

(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Here we have divided train test by 255 in order to bring them on 0-1 scale
X_train = X_train.reshape(60000, 784).astype('float32') / 255
X_test = X_test.reshape(10000, 784).astype('float32') / 255

# vanila model

def make_model(optimizer='adam', hidden_size=32):
    model = Sequential([
        Dense(hidden_size, input_shape=[784,]),
        Activation('relu'),
        Dense(hidden_size),
        Activation('relu'),
        Dense(10),
        Activation('softmax')])
        
    model.compile(optimizer=optimizer, loss = 'categorical_crossentropy', metrics=['accuracy'])
    return model

clf = KerasClassifier(make_model)

# fitting Multi-layer perceptron to the data using keras

num_classes=10
y_test = keras.utils.to_categorical(y_test, num_classes)


vanila_model=clf.fit(X_train, y_train, validation_data=(X_test, y_test),epochs=20, verbose=1)

# Visulize the vanila_model
# summarize history for accuracy
plt.plot(vanila_model.history['acc'])
plt.plot(vanila_model.history['val_acc'])
plt.title('vanila_model using traditional Train test split : learning curve')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['Vanila_train', 'Vanila_test'], loc='upper left')
plt.show()

# lets evaluate on test data
# as we see no significant gain after 5 epoch we check the score on test dataset

vanila_model = make_model()

num_classes=10
y_train = keras.utils.to_categorical(y_train, num_classes)

vanila_model.fit(X_train, y_train,epochs=5)

# model evaluation  on test data
np.random.RandomState(seed=0)
from keras.datasets import mnist

(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Here we have divided train test by 255 in order to bring them on 0-1 scale
X_train = X_train.reshape(60000, 784).astype('float32') / 255
X_test = X_test.reshape(10000, 784).astype('float32') / 255

num_classes=10
y_test = keras.utils.to_categorical(y_test, num_classes)

score = vanila_model.evaluate(X_test, y_test, verbose=0)

print("\nTest loss: {:.3f}".format(score[0])) # Test loss: 0.120
print("Test Accuracy: {:.3f}".format(score[1])) # Test Accuracy: 0.964

# saving the model if you do not want to train model again which usually saves time

# serialize model to JSON
model_json = vanila_model.to_json()
with open("Task2_model31.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
vanila_model.save_weights("Task2_model31.h5")
print("Saved model to disk")


# to load the saved model
'''
# load json and create model
json_file = open('Task2_model31.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("Task2_model31.h5")
print("Loaded model from disk")

# prediction using the stored model

# evaluate loaded model on test data
loaded_model.compile(optimizer='adam', loss = 'categorical_crossentropy', metrics=['accuracy'])
loaded_model_score = loaded_model.evaluate(X_test, y_test, verbose=0)
print("\nTest loss: {:.3f}".format(loaded_model_score[0])) # Test loss: 0.120
print("Test Accuracy: {:.3f}".format(loaded_model_score[1])) # Test Accuracy: 0.964
'''

# with dropout option

def make_model(optimizer='adam', hidden_size=32, dropout=0.2):
    model = Sequential([
        Dense(hidden_size, input_shape=[784,]),
        Activation('relu'),
        Dropout(rate = dropout),
        Dense(hidden_size),
        Activation('relu'),
        Dropout(rate = dropout),
        Dense(10),
        Activation('softmax')
    ])
        
    model.compile(optimizer=optimizer, loss = 'categorical_crossentropy', metrics=['accuracy'])
    return model

clf = KerasClassifier(make_model)


model_with_dropout=clf.fit(X_train, y_train, validation_data=(X_test, y_test),epochs=20, verbose=1)

# Visulize the model_with_dropout
# summarize history for accuracy
plt.plot(model_with_dropout.history['acc'])
plt.plot(model_with_dropout.history['val_acc'])
plt.title('model_with_dropout using traditional Train test split : learning curve')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['dropout_train', 'dropout_test'], loc='upper left')
plt.show()

# lets evaluate on test data
# as we see no significant gain after 5 epoch we check the score on test dataset

model_with_dropout = make_model()

num_classes=10
y_train = keras.utils.to_categorical(y_train, num_classes)

model_with_dropout.fit(X_train, y_train,epochs=5)

# model evaluation  on test data
np.random.RandomState(seed=0)
from keras.datasets import mnist

(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Here we have divided train test by 255 in order to bring them on 0-1 scale
X_train = X_train.reshape(60000, 784).astype('float32') / 255
X_test = X_test.reshape(10000, 784).astype('float32') / 255

num_classes=10
y_test = keras.utils.to_categorical(y_test, num_classes)

score = model_with_dropout.evaluate(X_test, y_test, verbose=0)

print("\nTest loss: {:.3f}".format(score[0])) # Test loss: 0.151
print("Test Accuracy: {:.3f}".format(score[1])) # Test Accuracy: 0.953

# saving the model if you do not want to train model again which usually saves time

# serialize model to JSON
model_json = model_with_dropout.to_json()
with open("Task2_model32.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model_with_dropout.save_weights("Task2_model32.h5")
print("Saved model to disk")


# to load the saved model
'''
# load json and create model
json_file = open('Task2_model32.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("Task2_model32.h5")
print("Loaded model from disk")

# prediction using the stored model

# evaluate loaded model on test data
loaded_model.compile(optimizer='adam', loss = 'categorical_crossentropy', metrics=['accuracy'])
loaded_model_score = loaded_model.evaluate(X_test, y_test, verbose=0)
print("\nTest loss: {:.3f}".format(loaded_model_score[0])) # Test loss: 0.151
print("Test Accuracy: {:.3f}".format(loaded_model_score[1])) # Test Accuracy: 0.953
'''

# To compare both model visuly we re load the models with diffrent epoch as those were got vanished in previous runs

np.random.RandomState(seed=0)
from keras.datasets import mnist

(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Here we have divided train test by 255 in order to bring them on 0-1 scale
X_train = X_train.reshape(60000, 784).astype('float32') / 255
X_test = X_test.reshape(10000, 784).astype('float32') / 255

# vanila model

def make_model(optimizer='adam', hidden_size=32):
    model = Sequential([
        Dense(hidden_size, input_shape=[784,]),
        Activation('relu'),
        Dense(hidden_size),
        Activation('relu'),
        Dense(10),
        Activation('softmax')])
        
    model.compile(optimizer=optimizer, loss = 'categorical_crossentropy', metrics=['accuracy'])
    return model

clf = KerasClassifier(make_model)

# fitting Multi-layer perceptron to the data using keras

num_classes=10
y_test = keras.utils.to_categorical(y_test, num_classes)


vanila_model=clf.fit(X_train, y_train, validation_data=(X_test, y_test),epochs=20, verbose=1)

# dropout model

def make_model(optimizer='adam', hidden_size=32, dropout=0.2):
    model = Sequential([
        Dense(hidden_size, input_shape=[784,]),
        Activation('relu'),
        Dropout(rate = dropout),
        Dense(hidden_size),
        Activation('relu'),
        Dropout(rate = dropout),
        Dense(10),
        Activation('softmax')
    ])
        
    model.compile(optimizer=optimizer, loss = 'categorical_crossentropy', metrics=['accuracy'])
    return model

clf = KerasClassifier(make_model)


model_with_dropout=clf.fit(X_train, y_train, validation_data=(X_test, y_test),epochs=20, verbose=1)

# to see both the model performance graphically 

plt.plot(vanila_model.history['acc'])
plt.plot(vanila_model.history['val_acc'])
plt.plot(model_with_dropout.history['acc'])
plt.plot(model_with_dropout.history['val_acc'])
plt.title('compare vanila model with dropout model : learning curve')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['Vanila_train', 'Vanila_test','dropout_train', 'dropout_test'], loc='bottom right')
plt.show()



