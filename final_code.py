import numpy as np
import matplotlib.pyplot as plt

from PIL import Image
import os
import pandas as pd

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
# from tensorflow.keras.utils import to_categorical
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Conv2D, MaxPool2D, Dense, Flatten, Dropout

from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from keras.optimizers import Adam

##Function to load the dataset
def load_dataset(classes,cur_path,data_folder):

    data=[]
    labels=[]

    for i in range(classes):
        path = os.path.join(cur_path,data_folder,'Train',str(i))
        images = os.listdir(path)

        for a in images:
            try:
                image = Image.open(path + '\\'+ a)
                image = image.resize((30,30))
                image = np.array(image)
                data.append(image)
                labels.append(i)
            except:
                print("Error loading image")

    data = np.array(data)
    labels=np.array(labels)

    return data,labels

##Function to pre-process the data.
##Min-Max normalization keeps the data between 0 and 1
def normalizer(data):
    scaler = MinMaxScaler()
    ascolumns = data.reshape(-1, 3)
    t = scaler.fit_transform(ascolumns)
    data = t.reshape(data.shape)
    return data

# Splitting training and validation dataset
def ttsplit(data,labels)->tuple:

    X_train, X_val, y_train, y_val = train_test_split(data, labels, test_size=0.2, random_state=42)
    y_train = to_categorical(y_train, 43)
    y_val = to_categorical(y_val, 43)
    return X_train,X_val,y_train,y_val

##The function to train the model
##Conv. layer- Reduces the dimensionality of the input image and helps idenify features
##Pooling layer- Resluts from conv layer are aggregated. This gives us the location of certain aspects of an image
##Flatten layer- The output from pooling layer is passed and we get a 1D array.
##Dense layer- Here is where the computation and classification happens. This is akin to our traditional neural network
##We have a set of input units which take data from the flattened layer
##Dropout layer- Reduces overfitting by dropping some of the connections between some units.
def model_training(new_model,X_train,y_train,X_val,y_val,epochs=1,batch_size=1):
    new_model.add(Conv2D(filters=27, kernel_size=(5,5), activation='relu', input_shape=X_train.shape[1:]))
    new_model.add(MaxPool2D(pool_size=(2, 2)))
    new_model.add(Flatten())
    new_model.add(Dense(150, activation='relu'))
    new_model.add(Dropout(rate=0.35))
    new_model.add(Dense(43, activation='softmax'))
    new_model.summary()

    # opt=Adam(learning_rate=0.1)
    new_model.compile(loss='categorical_crossentropy', optimizer="adam", metrics=['accuracy'])
    # new_model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    history = new_model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, validation_data=(X_val, y_val))
    new_model.save('traffic_classifier.h5')

    return history

#Plotting graphs for accuracy and loss
def plots(history,fig_no,hist_col1,l1,hist_col2,title,x_label,y_label):

    plt.figure(fig_no)
    plt.plot(history.history[hist_col1], label=l1)
    plt.plot(history.history[hist_col2], label=hist_col2)
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.legend()
    plt.show()

## Function for testing the model
def test(cur_path,data_folder,new_model,items=0):

    path = os.path.join(cur_path,data_folder)
    y_test = pd.read_csv(data_folder+'/'+'Test.csv')
    labels = y_test["ClassId"].values
    imgs = y_test["Path"].values

    if(items):
        new_imgs=[]
        new_label=[]
        for i in range(items):
            new_imgs.append(imgs[i])
            new_label.append(labels[i])
        imgs=new_imgs
        labels=new_label
    data=[]
    for img in imgs:
        image = Image.open(path + '\\' + img)
        image = image.resize((30,30))
        data.append(np.array(image))
    X_test = np.array(data)
    X_test=normalizer(X_test)
    pred = np.argmax(new_model.predict(X_test),axis=1)
    return pred,labels

## Function to evaluate performance
def evaluate(labels,pred):

    cm = confusion_matrix(labels, pred)
    print('Confusion Matrix:')
    print(cm)
    # accuracy: (tp + tn) / (p + n)
    accuracy = accuracy_score(labels, pred)
    # precision tp / (tp + fp)
    precision = precision_score(labels, pred, average='macro')
    # recall: tp / (tp + fn)
    recall = recall_score(labels, pred, average='macro')
    # f1: 2 tp / (2 tp + fp + fn)
    f1 = f1_score(labels, pred, average='macro')

    return accuracy,precision,recall,f1

data = []
labels = []
classes = 43
cur_path = os.getcwd()
data_folder = 'data'

new_data,new_labels=load_dataset(classes,cur_path,data_folder)
print(new_data)
print(new_labels)
scaled_data=normalizer(new_data)
print(scaled_data)
X_train,X_val,y_train,y_val=ttsplit(scaled_data,new_labels)
model=Sequential()
history=model_training(model,X_train,y_train,X_val,y_val,8,32)
plots(history,0,"accuracy","training accuracy","val_accuracy","Accuracy","epochs","accuracy")
plots(history,1,"loss","training loss","val_loss","Loss","epoch","loss")
# pred,test_labels=test(cur_path,data_folder,model,5)
# print("Predicted Outputs:")
# print(pred)
# print("Labels in the test_set")
# print(test_labels)
pred,test_labels=test(cur_path,data_folder,model)
accuracy,precision,recall,f1=evaluate(pred,test_labels)
print('Accuracy: %f' % accuracy)
print('Precision: %f' % precision)
print('Recall: %f' % recall)
print('F1 score: %f' % f1)