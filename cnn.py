from keras.layers import Merge,Convolution2D,MaxPooling2D,Dense,Flatten,Activation
from keras.models import Sequential
import cPickle
import numpy

if __name__ == '__main__':
    left_branch=Sequential()
    left_branch.add(Convolution2D(32,2,99,input_shape=(1,100,100)))
    left_branch.add(Activation("relu"))
    left_branch.add(MaxPooling2D(pool_size=(2,2)))
    left_branch.add(Activation("relu"))
    left_branch.add(Flatten())

    right_branch=Sequential()
    right_branch.add(Convolution2D(32,2,99,input_shape=(1,100,100,)))
    left_branch.add(Activation("relu"))
    right_branch.add(MaxPooling2D(pool_size=(2,2)))
    left_branch.add(Activation("relu"))
    right_branch.add(Flatten())

    merged=Merge([left_branch,right_branch],mode="concat")
    final_model=Sequential()
    final_model.add(merged)
    final_model.add(Dense(10))

    left_branch.compile(optimizer="rmsprop",loss="sparse_categorical_crossentropy",metrics=["accuracy"])
    right_branch.compile(optimizer="rmsprop",loss="sparse_categorical_crossentropy",metrics=["accuracy"])
    final_model.compile(optimizer="rmsprop",loss="sparse_categorical_crossentropy",metrics=["accuracy"])
    f=open("arg1_image_100","rb")
    arg1=cPickle.load(f)
    f=open("arg2_image_100","rb")
    arg2=cPickle.load(f)
    f=open("label","rb")
    label=cPickle.load(f)

    left_branch.fit(arg1,label)
    right_branch.fit(arg2,label)
    # todo
    final_model.fit(None,None,validation_split=0.1)
