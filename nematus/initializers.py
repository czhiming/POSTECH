#-*- coding:utf8 -*-
'''
Created on Apr 30, 2017
@function: 参数的初始化
@author: czm
'''
import numpy
import theano
import theano.tensor as tensor

def ortho_weight(ndim):
    W = numpy.random.randn(ndim,ndim)
    u,s,v = numpy.linalg.svd(W)
    return u.astype('float32')

def norm_weight(nin,nout=None,scale=0.01,ortho=True):
    if nout is None:
        nout = nin
    if nout == nin and ortho:
        W = ortho_weight(nin)
    else:
        W = scale*numpy.random.randn(nin,nout)
    return W.astype('float32')


if __name__ == '__main__':
    pass