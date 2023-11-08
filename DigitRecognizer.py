# basic
import numpy as np
import pandas as pd

# visuals
import matplotlib.pyplot as plt
import seaborn as sns

# Scikit-learn
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix,ConfusionMatrixDisplay,classification_report

# tensorflow
from keras.models import Sequential
from keras.layers import Conv2D,Dense,Flatten,MaxPooling2D,BatchNormalization,Dropout,SpatialDropout2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.utils import to_categorical
from tensorflow.keras.callbacks import ReduceLROnPlateau,EarlyStopping

labels = ['Zero','One','Two', 'Three', 'Four', 'Five', 'Six', 'Seven', 'Eight', 'Nine']

# train and test data import
train = pd.read_csv(r'D:\WorkSpace\Kaggle\DigitRecognizer\data\InputData\train.csv')
test = pd.read_csv(r'D:\WorkSpace\Kaggle\DigitRecognizer\data\InputData\test.csv')

# print the shape of train and test data

# print('The shape of train and test data is : ',train.shape,test.shape)

plt.figure(figsize=(20, 10))  # specifying the overall grid size
# plt.subplots_adjust(hspace=0.4)

train_labels = train['label'].values
train_images = train.drop('label', axis=1).values.reshape(-1, 28, 28, 1).astype(np.float32) / 255.0
test_images = test.values.reshape(-1, 28, 28, 1).astype(np.float32) / 255.0

# labels = ['Zero', 'One', 'Two', 'Three', 'Four', 'Five', 'Six', 'Seven', 'Eight', 'Nine']
#
# for i in range(50):
#     plt.subplot(5, 10, i + 1)  # the number of images in the grid is 5*5 (25)
#     plt.imshow(train_images[i])
#     plt.title(labels[int(train_labels[i])], fontsize=13)
#     # plt.title(labels[i])
#     plt.axis('off')
#
# plt.show()

X = train.drop(columns='label')
y = train['label']

X_train, X_, y_train, y_ = train_test_split(X, y, test_size=0.2, random_state=42)

X_val, X_test, y_val, y_test = train_test_split(X_, y_, test_size=0.2, random_state=42)

y_true = y_test

# function to change the data type and normalize the data and reshape the data.
def reshape(data):
    return data.values.reshape(-1, 28, 28, 1).astype(np.float32) / 255.0


# function to chnage the labels into categorical data
def to_cat(data):
    return to_categorical(data,num_classes=10)

X_train = reshape(X_train)
X_test = reshape(X_test)
X_val = reshape(X_val)
y_train = to_cat(y_train)
y_test = to_cat(y_test)
y_val = to_cat(y_val)

model = Sequential()

model.add(Conv2D(32,3,activation='relu',padding='same',input_shape=(28,28,1)))
#model.add(BatchNormalization())
model.add(Conv2D(32,3, activation = 'relu', padding ='same'))
#model.add(Dropout(0.2))
model.add(MaxPooling2D(2))


model.add(Conv2D(64,3,activation='relu',padding='same'))
#model.add(BatchNormalization())
model.add(Conv2D(64,3, activation = 'relu', padding ='same'))
model.add(MaxPooling2D(2))
#model.add(Dropout(0.2))

model.add(Conv2D(128, 3, padding = 'same', activation = 'relu'))
model.add(Conv2D(128, 3, padding = 'same', activation = 'relu'))
model.add(Flatten())

model.add(Dense(512,activation='relu'))
model.add(Dropout(0.2))
#model.add(Dense(128,activation='relu'))

model.add(Dense(10,activation='softmax'))

model.summary()

learning_rate_reduction = ReduceLROnPlateau(monitor = 'val_accuracy',
                                            patience=2,
                                            factor=0.5,
                                            min_lr = 0.00001,
                                            verbose = 1)

early_stoping = EarlyStopping(monitor='val_loss',patience= 5,restore_best_weights=True,verbose=0)

model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])

digit = model.fit(X_train,y_train,
                  validation_data=(X_val,y_val),
                  callbacks=[learning_rate_reduction,early_stoping],
                  batch_size = 10,
                  epochs = 30,
                  verbose=1
                 )


# Evaluvate for train generator
loss,acc = model.evaluate(X_train,y_train,batch_size = 10, verbose = 0)

print('The accuracy of the model for training data is:',acc*100)
print('The Loss of the model for training data is:',loss)

# Evaluvate for validation generator
loss,acc = model.evaluate(X_val,y_val,batch_size = 10, verbose = 0)

print('The accuracy of the model for validation data is:',acc*100)
print('The Loss of the model for validation data is:',loss)

# plots for accuracy and Loss with epochs

error = pd.DataFrame(digit.history)

plt.figure(figsize=(18,5),dpi=200)
sns.set_style('darkgrid')

plt.subplot(121)
plt.title('Cross Entropy Loss',fontsize=15)
plt.xlabel('Epochs',fontsize=12)
plt.ylabel('Loss',fontsize=12)
plt.plot(error['loss'])
plt.plot(error['val_loss'])

plt.subplot(122)
plt.title('Classification Accuracy',fontsize=15)
plt.xlabel('Epochs',fontsize=12)
plt.ylabel('Accuracy',fontsize=12)
plt.plot(error['accuracy'])
plt.plot(error['val_accuracy'])

plt.show()

result = model.predict(X_test, batch_size = 10,verbose = 0)

y_pred = np.argmax(result, axis = 1)

# Evaluvate
loss,acc = model.evaluate(X_test,y_test, batch_size = 10, verbose = 0)

print('The accuracy of the model for testing data is:',acc*100)
print('The Loss of the model for testing data is:',loss)


print(classification_report(y_true, y_pred,target_names=labels))

confusion_mtx = confusion_matrix(y_true,y_pred)

sns.set_style('ticks')
f,ax = plt.subplots(figsize = (20,8),dpi=200)
sns.heatmap(confusion_mtx, annot=True, linewidths=0.1, cmap = "gist_yarg_r", linecolor="black", fmt='.0f', ax=ax,cbar=False, xticklabels=labels, yticklabels=labels)

plt.xlabel("Predicted Label",fontsize=10)
plt.ylabel("True Label",fontsize=10)
plt.title("Confusion Matrix",fontsize=13)

plt.show()


# first pre-process the test data
test_data = reshape(test)

# prediction
pred = model.predict(test_data, batch_size = 10,verbose = 0)

prediction = np.argmax(pred, axis = 1)

# submission
submission = pd.read_csv('D:\WorkSpace\Kaggle\DigitRecognizer\data\InputData\sample_submission.csv')

submission['Label'] = prediction

submission.head()

submission.to_csv('D:\WorkSpace\Kaggle\DigitRecognizer\data\OutputData\sample_submission.csv',index=False)