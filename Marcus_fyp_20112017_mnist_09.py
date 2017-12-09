from keras.models import Sequential
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.optimizers import SGD,adam, RMSprop
from keras.utils import np_utils
import os
import matplotlib.pyplot as plt
import numpy as np
import cv2
import sklearn 
import tensorflow
from sklearn.model_selection import train_test_split
from PIL import Image

np.set_printoptions(threshold=np.inf)

'''
def shuffle(train_data, num_sample):
	seed=np.random.random(1000)
	
	for i in range (0:num_sample):
'''

#system parameters
batch_size=50			#size of each batch
nb_classes=10			#number of output state
nb_epoch=20				#number of time training through the set of data
nb_filters=32			#number of filters for convolution
nb_pool=2				#pooling size
nb_conv=2				#convolution kernel size

#0
path1=r"C:\Users\justBaloney\Pictures\FYP Database\Mnist\0"	#path of original data
path2=r"C:\Users\justBaloney\Pictures\FYP Database\Mnist\0_output"	#path to store pre-processed data
print(path1)
listing=os.listdir(path1)
num_sample=len(listing)
img_row, img_column=20,20

#resizing, and gray-scaling the original images
for file in listing:
	img=Image.open(path1+'\\'+file)
	im=img.resize((img_row, img_column))
	gray=im.convert('L')
	gray.save(path2 + '\\' + file,"JPEG")
	#im = Image.open(path1 + '\\' + file)
    #img = im.resize((img_row, img_column))
    #gray = img.convert('L')
    # need to do some more processing here
    #gray.save(path2 + '\\' + file, "JPEG")


#storing the images in a matrix
data_matrix_0=np.array([np.array(Image.open(path2+'\\'+file)).flatten()
						for file in listing],'f')

#1
path1=r"C:\Users\justBaloney\Pictures\FYP Database\Mnist\1"	#path of original data
path2=r"C:\Users\justBaloney\Pictures\FYP Database\Mnist\1_output"	#path to store pre-processed data
print(path1)
listing=os.listdir(path1)
num_sample=len(listing)
img_row, img_column=20,20

#resizing, and gray-scaling the original images
for file in listing:
	img=Image.open(path1+'\\'+file)
	im=img.resize((img_row, img_column))
	gray=im.convert('L')
	gray.save(path2 + '\\' + file,"JPEG")
	#im = Image.open(path1 + '\\' + file)
    #img = im.resize((img_row, img_column))
    #gray = img.convert('L')
    # need to do some more processing here
    #gray.save(path2 + '\\' + file, "JPEG")

data_matrix_1=np.array([np.array(Image.open(path2+'\\'+file)).flatten()
						for file in listing],'f')

#2
path1=r"C:\Users\justBaloney\Pictures\FYP Database\Mnist\2"	#path of original data
path2=r"C:\Users\justBaloney\Pictures\FYP Database\Mnist\2_output"	#path to store pre-processed data
print(path1)
listing=os.listdir(path1)
num_sample=len(listing)
img_row, img_column=20,20

#resizing, and gray-scaling the original images
for file in listing:
	img=Image.open(path1+'\\'+file)
	im=img.resize((img_row, img_column))
	gray=im.convert('L')
	gray.save(path2 + '\\' + file,"JPEG")
	#im = Image.open(path1 + '\\' + file)
    #img = im.resize((img_row, img_column))
    #gray = img.convert('L')
    # need to do some more processing here
    #gray.save(path2 + '\\' + file, "JPEG")

data_matrix_2=np.array([np.array(Image.open(path2+'\\'+file)).flatten()
						for file in listing],'f')

#3
path1=r"C:\Users\justBaloney\Pictures\FYP Database\Mnist\3"	#path of original data
path2=r"C:\Users\justBaloney\Pictures\FYP Database\Mnist\3_output"	#path to store pre-processed data
print(path1)
listing=os.listdir(path1)
num_sample=len(listing)
img_row, img_column=20,20

#resizing, and gray-scaling the original images
for file in listing:
	img=Image.open(path1+'\\'+file)
	im=img.resize((img_row, img_column))
	gray=im.convert('L')
	gray.save(path2 + '\\' + file,"JPEG")
	#im = Image.open(path1 + '\\' + file)
    #img = im.resize((img_row, img_column))
    #gray = img.convert('L')
    # need to do some more processing here
    #gray.save(path2 + '\\' + file, "JPEG")

data_matrix_3=np.array([np.array(Image.open(path2+'\\'+file)).flatten()
						for file in listing],'f')

#4
path1=r"C:\Users\justBaloney\Pictures\FYP Database\Mnist\4"	#path of original data
path2=r"C:\Users\justBaloney\Pictures\FYP Database\Mnist\4_output"	#path to store pre-processed data
print(path1)
listing=os.listdir(path1)
num_sample=len(listing)
img_row, img_column=20,20

#resizing, and gray-scaling the original images
for file in listing:
	img=Image.open(path1+'\\'+file)
	im=img.resize((img_row, img_column))
	gray=im.convert('L')
	gray.save(path2 + '\\' + file,"JPEG")
	#im = Image.open(path1 + '\\' + file)
    #img = im.resize((img_row, img_column))
    #gray = img.convert('L')
    # need to do some more processing here
    #gray.save(path2 + '\\' + file, "JPEG")

data_matrix_4=np.array([np.array(Image.open(path2+'\\'+file)).flatten()
						for file in listing],'f')

#5
path1=r"C:\Users\justBaloney\Pictures\FYP Database\Mnist\5"	#path of original data
path2=r"C:\Users\justBaloney\Pictures\FYP Database\Mnist\5_output"	#path to store pre-processed data
print(path1)
listing=os.listdir(path1)
num_sample=len(listing)
img_row, img_column=20,20

#resizing, and gray-scaling the original images
for file in listing:
	img=Image.open(path1+'\\'+file)
	im=img.resize((img_row, img_column))
	gray=im.convert('L')
	gray.save(path2 + '\\' + file,"JPEG")
	#im = Image.open(path1 + '\\' + file)
    #img = im.resize((img_row, img_column))
    #gray = img.convert('L')
    # need to do some more processing here
    #gray.save(path2 + '\\' + file, "JPEG")

data_matrix_5=np.array([np.array(Image.open(path2+'\\'+file)).flatten()
						for file in listing],'f')

#6
path1=r"C:\Users\justBaloney\Pictures\FYP Database\Mnist\6"	#path of original data
path2=r"C:\Users\justBaloney\Pictures\FYP Database\Mnist\6_output"	#path to store pre-processed data
print(path1)
listing=os.listdir(path1)
num_sample=len(listing)
img_row, img_column=20,20

#resizing, and gray-scaling the original images
for file in listing:
	img=Image.open(path1+'\\'+file)
	im=img.resize((img_row, img_column))
	gray=im.convert('L')
	gray.save(path2 + '\\' + file,"JPEG")
	#im = Image.open(path1 + '\\' + file)
    #img = im.resize((img_row, img_column))
    #gray = img.convert('L')
    # need to do some more processing here
    #gray.save(path2 + '\\' + file, "JPEG")

data_matrix_6=np.array([np.array(Image.open(path2+'\\'+file)).flatten()
						for file in listing],'f')

#7
path1=r"C:\Users\justBaloney\Pictures\FYP Database\Mnist\7"	#path of original data
path2=r"C:\Users\justBaloney\Pictures\FYP Database\Mnist\7_output"	#path to store pre-processed data
print(path1)
listing=os.listdir(path1)
num_sample=len(listing)
img_row, img_column=20,20

#resizing, and gray-scaling the original images
for file in listing:
	img=Image.open(path1+'\\'+file)
	im=img.resize((img_row, img_column))
	gray=im.convert('L')
	gray.save(path2 + '\\' + file,"JPEG")
	#im = Image.open(path1 + '\\' + file)
    #img = im.resize((img_row, img_column))
    #gray = img.convert('L')
    # need to do some more processing here
    #gray.save(path2 + '\\' + file, "JPEG")

data_matrix_7=np.array([np.array(Image.open(path2+'\\'+file)).flatten()
						for file in listing],'f')

#8
path1=r"C:\Users\justBaloney\Pictures\FYP Database\Mnist\8"	#path of original data
path2=r"C:\Users\justBaloney\Pictures\FYP Database\Mnist\8_output"	#path to store pre-processed data
print(path1)
listing=os.listdir(path1)
num_sample=len(listing)
img_row, img_column=20,20

#resizing, and gray-scaling the original images
for file in listing:
	img=Image.open(path1+'\\'+file)
	im=img.resize((img_row, img_column))
	gray=im.convert('L')
	gray.save(path2 + '\\' + file,"JPEG")
	#im = Image.open(path1 + '\\' + file)
    #img = im.resize((img_row, img_column))
    #gray = img.convert('L')
    # need to do some more processing here
    #gray.save(path2 + '\\' + file, "JPEG")

data_matrix_8=np.array([np.array(Image.open(path2+'\\'+file)).flatten()
						for file in listing],'f')

#9
path1=r"C:\Users\justBaloney\Pictures\FYP Database\Mnist\9"	#path of original data
path2=r"C:\Users\justBaloney\Pictures\FYP Database\Mnist\9_output"	#path to store pre-processed data
print(path1)
listing=os.listdir(path1)
num_sample=len(listing)
img_row, img_column=20,20

#resizing, and gray-scaling the original images
for file in listing:
	img=Image.open(path1+'\\'+file)
	im=img.resize((img_row, img_column))
	gray=im.convert('L')
	gray.save(path2 + '\\' + file,"JPEG")
	#im = Image.open(path1 + '\\' + file)
    #img = im.resize((img_row, img_column))
    #gray = img.convert('L')
    # need to do some more processing here
    #gray.save(path2 + '\\' + file, "JPEG")

data_matrix_9=np.array([np.array(Image.open(path2+'\\'+file)).flatten()
						for file in listing],'f')

data_matrix=np.concatenate((data_matrix_0,data_matrix_1,data_matrix_2,data_matrix_3,data_matrix_4, data_matrix_5,
							data_matrix_6,data_matrix_7,data_matrix_8, data_matrix_9),axis=0)

load_path=path2
listing=os.listdir(load_path)
print("The total number of data images is:\n")
num_sample=(len(os.listdir(r"C:\Users\justBaloney\Pictures\FYP Database\Mnist\0_output"))+
			len(os.listdir(r"C:\Users\justBaloney\Pictures\FYP Database\Mnist\1_output"))+
			len(os.listdir(r"C:\Users\justBaloney\Pictures\FYP Database\Mnist\2_output"))+
			len(os.listdir(r"C:\Users\justBaloney\Pictures\FYP Database\Mnist\3_output"))+
			len(os.listdir(r"C:\Users\justBaloney\Pictures\FYP Database\Mnist\4_output"))+
			len(os.listdir(r"C:\Users\justBaloney\Pictures\FYP Database\Mnist\5_output"))+
			len(os.listdir(r"C:\Users\justBaloney\Pictures\FYP Database\Mnist\6_output"))+
			len(os.listdir(r"C:\Users\justBaloney\Pictures\FYP Database\Mnist\7_output"))+
			len(os.listdir(r"C:\Users\justBaloney\Pictures\FYP Database\Mnist\8_output"))+
			len(os.listdir(r"C:\Users\justBaloney\Pictures\FYP Database\Mnist\9_output")))
print(num_sample)

input_data=np.array(data_matrix)
print(data_matrix)

#labeling of image: 0:0	1:1 2:2 3:3 4:4 5:5 6:6 7:7 8:8 9:9
label=np.ones((num_sample),dtype=int)
label[0:500]=0
label[500:1000]=1
label[1000:1500]=2
label[1500:2000]=3
label[2000:2500]=4
label[2500:3000]=5
label[3000:3500]=6
label[3500:4000]=7
label[4000:4500]=8
label[4500:5000]=9
print (label)

#shuffling data set
data, Label=sklearn.utils.shuffle(data_matrix, label, random_state=1)
train_data=[data, Label]
np.set_printoptions(threshold=10)
print (train_data[0])
print(train_data[1])

#passing of data and labeling to variable X and y
(X,y)=(train_data[0],train_data[1])

#splitting the data set into training and testng sets
X_train,X_test,y_train,y_test=train_test_split(X,y, test_size=0.2, random_state=4)
X_train=X_train.reshape(X_train.shape[0],img_row,img_column, 1)
X_test=X_test.reshape(X_test.shape[0],img_row,img_column,1)

X_train=X_train.astype('float32')
X_test=X_test.astype('float32')

X_train/=255
X_test/=255

print("X_train shape:", X_train.shape)
print("Now print the size of the input data:\n\n")
print(X_train.shape[0]," training sample")
print(X_test.shape[0]," testing sample")

print("\nNow print the actual training and testing data:\n")

print("Column: ", X_train.shape[2])
print("Row: ", X_train.shape[1])

'''
print(X_test)
print(y_train)
print(y_test)
'''

#conversion of class vector to binary class
Y_train=np_utils.to_categorical(y_train,nb_classes)
#Y_train=Y_train.reshape((-1,1))
Y_test=np_utils.to_categorical(y_test,nb_classes)
#Y_test=Y_test.reshape((-1,1))
print("\nY_train: ",Y_train.shape)
print("\nY_test: ", Y_test.shape)

j=2
plt.imshow(X_train[j,0],interpolation='nearest')
print("label:",Y_train)
print("\nlabel_test:",Y_test)

#con. neural network architecture
model=Sequential()
model.add(Convolution2D(nb_filters,nb_conv,nb_conv, border_mode='valid',input_shape=(img_row,img_column,1)))
conv1_out=Activation('relu')
model.add(conv1_out)
model.add(Convolution2D(nb_filters,nb_conv,nb_conv))
model.add(Dropout(0.5))

conv2_out=Activation('relu')
model.add(conv2_out)
#model.add(Convolution2D(nb_filters,nb_classes,nb_conv))
#model.add(Dropout(0.5))

#conv3_out=Activation('relu')
#model.add(conv3_out)
model.add(MaxPooling2D(pool_size=(nb_pool,nb_pool)))
model.add(Dropout(0.5))

model.add(Flatten())
model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dropout(0.5))
#model.add(Dense(64))
#model.add(Activation('relu'))
#model.add(Dropout(0.5))
model.add(Dense(nb_classes))
#model.add(Flatten())
model.add(Activation('softmax'))
model.compile(metrics=['accuracy'],loss='categorical_crossentropy',optimizer='adam')
#END of con. neural network architecture

#training

#fitted_model=model.fit(X_train,	Y_train,	batch_size=batch_size,	nb_epoch=nb_epoch,	verbose=1,	validation_data=(X_test,Y_test))
#fitted_model=model.fit(X_train,	Y_train,	batch_size=batch_size,	nb_epoch=nb_epoch,	verbose=1,	validation_split=0.2)
hist = model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch,
                 verbose=1, validation_data=(X_test,Y_test))

hist = model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch,
                 verbose=1, validation_split=0.2)	#to change the validation_split value

#evaluation
score=model.evaluate(X_test,Y_test, verbose=2)
print("Test score:", score[0])
print("Test accuracy:", score[1])

'''
edit log
error message:
ValueError: Error when checking target: expected activation_4 to have shape (None, 2) but got array with shape (2, 1)
'''
