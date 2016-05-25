import numpy as np
import  word2vec
import codecs
import json
import dill
import cPickle

def loadArg1():
    model=word2vec.load("/mnt/mint_share/text8.bin")
    data=np.empty((17572,1,100,100),dtype='float64')
    label=np.empty((17472,),dtype='uint8')
    with codecs.open("/mnt/mint_share/train_pdtb.json","rU","utf-8") as f:
        for i,line in  enumerate(f):
            unit=json.loads(line)
            len1 = len(unit['Arg1']['Word'])
            if(len1 <100):
                for j in range(len1):
                    try:
                        j_ = model[unit['Arg1']['Word'][j]]
                    except:
                        j_ = model['fillin']
                    data[i,:,j,:]= j_
                for j in range(100- len1):
                    data[i,:,len1+j,:]=model['fillin']
            else:
                for j in range(100):
                    try:
                        j_ = model[unit['Arg1']['Word'][j]]
                    except:
                        j_ = model['fillin']
                    data[i,:,j,:]= j_
    with open("arg1_image_100","wb") as f1:
        # dill.dump(data,f1)
        cPickle.dump(data,f1,protocol=2)


def loadArg2():
    model=word2vec.load("/mnt/mint_share/text8.bin")
    data=np.empty((17572,1,100,100),dtype='float64')
    with codecs.open("/mnt/mint_share/train_pdtb.json","rU","utf-8") as f:
        for i,line in  enumerate(f):
            unit=json.loads(line)
            len1 = len(unit['Arg2']['Word'])
            if(len1 <100):
                for j in range(len1):
                    try:
                        j_ = model[unit['Arg2']['Word'][j]]
                    except:
                        j_ = model['fillin']
                    data[i,:,j,:]= j_
                for j in range(100- len1):
                    data[i,:,len1+j,:]=model['fillin']
            else:
                for j in range(100):
                    try:
                        j_ = model[unit['Arg2']['Word'][j]]
                    except:
                        j_ = model['fillin']
                    data[i,:,j,:]= j_
    with open("arg2_image_100","wb") as f1:
        cPickle.dump(data,f1,protocol=2)


if __name__ == '__main__':
    loadArg2()