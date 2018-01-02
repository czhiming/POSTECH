#-*- coding:utf8 -*-
'''
Created on Jun 16, 2017

@author: czm
'''
from RNN import data_iter
from nematus import nmt
from nematus import data_iterator
import theano
from theano import tensor
from nematus.util import load_config
from nematus.theano_util import load_params,init_theano_params
from RNN import rnn, rnn_trg, rnn_stack_pro
import numpy

def qe_rnn(model='RNN_model/wmt17.en-de.npz',
           datasets=['test/test16.bpe.en',
                     'test/test16.bpe.de'],
           save_file='test16.hter.pred'
           ):
    
    options = load_config(model)
    params = rnn.init_params(options)
    params = load_params(model, params)
    tparams = init_theano_params(params)
    
    trng,use_noise,x,x_mask,y,y_mask,\
        hter,y_pred,cost = rnn.build_model(tparams,options)
    inps = [x,x_mask,y,y_mask]
    f_pred = theano.function(inps,y_pred,profile=False)
    
    test = data_iterator.TextIterator(datasets[0],datasets[1],
                        options['dictionaries'][0], options['dictionaries'][1],
                        n_words_source=options['n_words_src'], n_words_target=options['n_words_tgt'],
                        batch_size=options['valid_batch_size'],
                        maxlen=options['maxlen'],
                        sort_by_length=False)

    res = []
    n_samples = 0
    
    for x,y in test:
        x, x_mask, y, y_mask = nmt.prepare_data(x, y, maxlen=options['maxlen'],
                                                n_words_src=options['n_words_src'],
                                                n_words=options['n_words_tgt'])
        
        res.extend(list(f_pred(x,x_mask,y,y_mask).flatten()))
        n_samples += x.shape[1]
        print 'processed:',n_samples,'samples'
    
    with open('qe/'+save_file,'w') as fp:
        for hh in res:
            fp.writelines(str(hh)+'\n')

def qe_rnn_stack_pro(model='stack_model/stack.en-de.npz',
           datasets=['test/test16.bpe.en',
                     'test/test16.bpe.de'],
           save_file='test16.hter.pred'
           ):
    
    options = load_config(model)
    params = rnn_stack_pro.init_params(options) # ******
    params = load_params(model, params)
    tparams = init_theano_params(params)
    
    trng,use_noise,x,x_mask,y,y_mask,\
        hter,y_pred,cost = rnn_stack_pro.build_model(tparams,options) # *****
    inps = [x,x_mask,y,y_mask]
    f_pred = theano.function(inps,y_pred,profile=False)
    
    test = data_iterator.TextIterator(datasets[0],datasets[1],
                        options['dictionaries'][0], options['dictionaries'][1],
                        n_words_source=options['n_words_src'], n_words_target=options['n_words_tgt'],
                        batch_size=options['valid_batch_size'],
                        maxlen=options['maxlen'],
                        sort_by_length=False)

    res = []
    n_samples = 0
    
    for x,y in test:
        x, x_mask, y, y_mask = nmt.prepare_data(x, y, maxlen=options['maxlen'],
                                                n_words_src=options['n_words_src'],
                                                n_words=options['n_words_tgt'])
        
        res.extend(list(f_pred(x,x_mask,y,y_mask).flatten()))
        n_samples += x.shape[1]
        print 'processed:',n_samples,'samples'
    
    with open('qe/'+save_file,'w') as fp:
        for hh in res:
            fp.writelines(str(hh)+'\n')

def qe_rnn_trg(model='RNN_trg_model/wmt17.de-en.npz',
           datasets=['test/test16.bpe.src',
                     'test/test16.bpe.mt'],
           save_file='test16.hter.pred'
           ):
    
    options = load_config(model)
    #-------------------
    params = rnn_trg.init_params(options)  # 修改此处
    params = load_params(model, params)
    tparams = init_theano_params(params)
    
    #-------------------
    trng,use_noise,x,x_mask,y,y_mask,\
        hter,y_pred,cost = rnn_trg.build_model(tparams,options) # 修改此处
    inps = [x,x_mask,y,y_mask]
    f_pred = theano.function(inps,y_pred,profile=False)
    
    test = data_iterator.TextIterator(datasets[0],datasets[1],
                        options['dictionaries'][0], options['dictionaries'][1],
                        n_words_source=options['n_words_src'], n_words_target=options['n_words_tgt'],
                        batch_size=options['valid_batch_size'],
                        maxlen=options['maxlen'],
                        sort_by_length=False)

    res = []
    n_samples = 0
    
    for x,y in test:
        x, x_mask, y, y_mask = nmt.prepare_data(x, y, maxlen=options['maxlen'],
                                                n_words_src=options['n_words_src'],
                                                n_words=options['n_words_tgt'])
        
        res.extend(list(f_pred(x,x_mask,y,y_mask).flatten()))
        n_samples += x.shape[1]
        print 'processed:',n_samples,'samples'
    
    with open('qe/'+save_file,'w') as fp:
        for hh in res:
            fp.writelines(str(hh)+'\n')


if __name__ == '__main__':
    pass
    
    
    
    