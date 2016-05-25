import codecs
import numpy as np
import json
import cPickle

if __name__ == '__main__':
    label=np.empty((17572,),dtype="uint8")
    senseDic={}
    senseList = [u'Expansion.List', u'Expansion.Conjunction', u'Expansion.Instantiation', u'Expansion.Alternative',
             u'EntRel', u'Contingency.Cause', u'Contingency.Pragmatic cause', u'Temporal.Asynchronous',
             u'Comparison.Contrast', u'Comparison.Concession', u'Expansion.Restatement', u'Temporal.Synchrony']
    for i, sense_item in enumerate(senseList):
        senseDic[sense_item] = i

    with codecs.open("/mnt/mint_share/train_pdtb.json","rU","utf-8") as f:
        for i,line in enumerate(f):
            unit=json.loads(line)
            sense=unit['Sense'][0]
            label[i]=senseDic[sense]

    with open("label","wb") as labelFile:
        cPickle.dump(label,labelFile)
