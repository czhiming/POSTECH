#-*- coding:utf8 -*-
'''
Created on Jun 15, 2017

@author: czm
'''
from nematus.nmt import prepare_data,build_model,init_params
from nematus.theano_util import init_theano_params,load_params
from nematus.util import load_config
from nematus.data_iterator import TextIterator
import theano
import sys
import numpy
import cPickle as pkl

def init():
    pass
    
def get_qv(model='model/model.npz.best_bleu'):
    """
    @function:获得质量向量(quality vector)
    """
    options = load_config(model)
    
    params = init_params(options)
    params = load_params(model, params)
    
    tparams = init_theano_params(params)
    
    trng,use_noise,x,x_mask,y,y_mask,\
        opt_ret, cost, ctx, tt = build_model(tparams,options)
    
    #加载数据
    train = TextIterator(options['datasets'][0], options['datasets'][1],
                            options['dictionaries'][0], options['dictionaries'][1],
                            n_words_source=options['n_words_src'], n_words_target=options['n_words_tgt'],
                            batch_size=options['batch_size'],
                            maxlen=1000, #设置尽可能长的长度
                            sort_by_length=False) #设为 False
    
    dev = TextIterator(options['valid_datasets'][0], options['valid_datasets'][1],
                            options['dictionaries'][0], options['dictionaries'][1],
                            n_words_source=options['n_words_src'], n_words_target=options['n_words_tgt'],
                            batch_size=options['valid_batch_size'],
                            maxlen=1000, #设置尽可能长的长度
                            sort_by_length=False) #设为 False
    
    
    f_tt = theano.function([x,x_mask,y,y_mask],tt,name='f_tt')
    
    #print tparams['ff_logit_W'].get_value().shape   #### (500,40000)
    n_samples = 0
    
    for x, y in train:
            # 准备数据用于训练
            x, x_mask, y, y_mask = prepare_data(x, y, maxlen=1000,
                                                n_words_src=options['n_words_src'],
                                                n_words=options['n_words_tgt'])
            tt_ = f_tt(x,x_mask,y,y_mask)
            Wt = tparams['ff_logit_W'].get_value()
            
            for j in range(y.shape[1]):
                
                qv_ = []
                for i in range(y.shape[0]):
                    if y_mask[i][j] == 1:
                        index = y[i][j]
                        qv = tt_[i,0,:].T*Wt[:,index]
                        qv_.append(list(qv))
            
                with open('qv/train/'+str(n_samples+j)+'.qv.pkl','w') as fp:
                    pkl.dump(qv_,fp)
                    
            n_samples += y.shape[1]
            print 'processed:',n_samples,'samples ...'
    
if __name__ == '__main__':
    get_qv()
    