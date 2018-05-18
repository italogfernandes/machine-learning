# Artificial Neural Network

# Installing Theano
# pip install --upgrade --no-deps git+git://github.com/Theano/Theano.git

# Installing Tensorflow
# Install Tensorflow from the website: https://www.tensorflow.org/versions/r0.12/get_started/os_setup.html

# Installing Keras
# pip install --upgrade keras

# Part 1 - Data Preprocessing

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('../datasets/wine_dataset.csv')
X = dataset.iloc[:, 1:].values
y = dataset.iloc[:, 0].values

t = np.zeros((178,3),dtype=int)
for i in range(1,4):
    t[:,i-1] = (y==i).astype(int) 

yy = y
y = t
# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Part 2 - Now let's make the ANN!

# Importing the Keras libraries and packages
import keras
from keras.models import Sequential
from keras.layers import Dense

# Initialising the ANN
classifier = Sequential()

# Adding the input layer and the first hidden layer
classifier.add(Dense(output_dim = 3, init = 'uniform', activation = 'relu', input_dim = 13))

# Adding the second hidden layer
#classifier.add(Dense(output_dim = 13, init = 'uniform', activation = 'relu'))

# Adding the output layer
classifier.add(Dense(output_dim = 3, init = 'uniform', activation = 'sigmoid'))

# Compiling the ANN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Fitting the ANN to the Training set
classifier.fit(X_train, y_train, batch_size = 10, nb_epoch = 100)

# Part 3 - Making the predictions and evaluating the model

# Predicting the Test set results
y_pred = classifier.predict(X_train)
#y_pred = (y_pred > 0.5)

# confusion matrix
#%%
def get_labeled_matriz(sparce):
    dense = np.zeros((sparce.shape[0],1))
    for i in range(sparce.shape[0]):
        ind = np.where(sparce[i,:] == (sparce[i,:].max()))[0][0]
        dense[i] = ind + 1
    return dense
#%%

#%% Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
y_pred_l = get_labeled_matriz(y_pred)
y_test_l = get_labeled_matriz(y_train)
cm = confusion_matrix(y_test_l, y_pred_l)

cole = np.append(arr=y_pred_l, values=y_test_l, axis=1)
#%%
lines = np.zeros((3,1))
columns = np.zeros((1,4))
total_acertos = 0
for i in range(3):
    lines[i,0] = cm[i,i] / np.sum(cm[i,:])
    columns[0,i] = cm[i,i] / np.sum(cm[:,i])
    total_acertos += cm[i,i]
t_acc = total_acertos / np.sum(cm)
columns[0,3] = t_acc
cm2 = np.append(arr=cm, values=lines, axis=1)
cm2 = np.append(arr=cm2, values=columns, axis=0)