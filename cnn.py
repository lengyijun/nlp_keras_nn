from keras.layers import Merge,Convolution2D,MaxPooling2D,Dense,Flatten,Activation,Input,merge,Dropout
from keras.layers.core import Activation
from keras.models import Sequential,Model
from keras.utils.visualize_util import plot
import cPickle
import numpy

if __name__ == '__main__':
    left_branch=Input(shape=(1,100,100,),name="arg1_input")
    left_branch_convo=(Convolution2D(32,4,99,))(left_branch)
    # left_branch_acti=(Activation("tanh"))(left_branch_convo)
    left_branch_pool=(MaxPooling2D(pool_size=(2,1)))(left_branch_convo)
    left_branch_flatten=(Flatten())(left_branch_pool)


    right_branch=Input(shape=(1,100,100,),name="arg2_input")
    right_branch_convo=(Convolution2D(32,4,99,))(right_branch)
    # right_branch_acti=(Activation("tanh"))(right_branch_convo)
    right_branch_pool=(MaxPooling2D(pool_size=(2,1)))(right_branch_convo)
    right_branch_flatten=(Flatten())(right_branch_pool)

    merged=merge([left_branch_flatten,right_branch_flatten],mode="concat")
    # merged_dropout=(Dropout(0.9))(merged)
    merged_dense=(Dense(12,name="cate"))(merged) #12 category

    final_model=Model(input=[left_branch,right_branch],output=[merged_dense])
    plot(final_model,to_file="model.png")

    # jsonfile=final_model.to_json()
    # with open("modeljson","w") as f :
    #     f.write(jsonfile)
    # f.close()

    f=open("arg1_image_100","rb")
    arg1=cPickle.load(f)
    f=open("arg2_image_100","rb")
    arg2=cPickle.load(f)
    f=open("label","rb")
    label=cPickle.load(f)
    f.close()


    final_model.compile(optimizer="rmsprop",loss="sparse_categorical_crossentropy",metrics=["accuracy"])
    final_model.fit({"arg1_input":arg1,"arg2_input":arg2},{"cate":label})
    final_model.save_weights("my_model_weights.h5")

    f=open("dev_arg1_image_100","rb")
    arg1=cPickle.load(f)
    f=open("dev_arg2_image_100","rb")
    arg2=cPickle.load(f)
    f=open("dev_label","rb")
    label=cPickle.load(f)
    f.close()

    print "="*9
    print final_model.evaluate({"arg1_input":arg1,"arg2_input":arg2},{"cate":label})
