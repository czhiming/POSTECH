#-*- coding:utf8 -*-
'''
Created on Aug 27, 2017

@author: czm
'''
from nematus.theano_util import concatenate

def get_qv_w2c(emb,y,w2v,dim=500):
    """
    @function:将词向量加入质量向量中
    """
    
    #model = gensim.models.KeyedVectors.load_word2vec_format('model/wmt17qe.vectors.de.500.tok.bin',binary=True)
    
    yshape = y.shape
    
    result = w2v[y.flatten()]
    result = result.reshape([yshape[0],yshape[1],dim])
    result = concatenate([emb,result],axis=2) #将两个张量进行拼接
    
    return result









if __name__ == '__main__':
    pass