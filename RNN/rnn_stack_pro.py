#-*- coding:utf8 -*-
'''
Created on Jun 16, 2017

@author: czm
'''
import theano
from theano import tensor
from nematus.util import load_dict,load_config
from RNN.data_iter import TextIterator
from collections import OrderedDict
from nematus.layers import param_init_gru,param_init_fflayer,shared_dropout_layer,gru_layer
from nematus.theano_util import load_params,init_theano_params,itemlist,unzip_from_theano,zip_to_theano
import numpy
import os
from nematus import nmt
from nematus import data_iterator
from nematus.optimizers import adadelta,rmsprop,adam
import time
import json
import ipdb
import copy
import sys
from word_embedding import get_qv_w2c
import gensim
from PIL.WmfImagePlugin import word
import rnn

def train(
    batch_size=80,
    valid_batch_size=80,
    dim=100,
    dim_word=500,
    dispFreq=100,
    saveFreq=3000,
    validFreq=1000,
    saveto='stack_model/stack.en-de.npz',
    datasets=['tuning/train.bpe.en',
              'tuning/train.bpe.de'],
    valid_datasets=['tuning/dev.bpe.en',
                    'tuning/dev.bpe.de'],
    dictionaries=['data/train.bpe.en.json',
                  'data/train.bpe.de.json'],
    hter=['tuning/train.hter',
          'tuning/dev.hter'],
    n_words_src=40000,
    n_words_tgt=40000,
    nmt_model='model/model.npz.best_bleu',
    final_model='RNN_model/wmt17.en-de.npz',
    lrate=0.0001,  # learning rate
    
    use_dropout=True,
    patience=10,
    max_epochs=5000,
    finish_after=1000000,
    maxibatch_size=20,
    optimizer='rmsprop',
    shuffle_each_epoch=True,
    reload_=True,
    overwrite=False,
    sort_by_length=False,
    maxlen=1000,
    decay_c=0.,  # L2 regularization penalty
    map_decay_c=0., # L2 regularization penalty towards original weights
    clip_c=1.0,
    dropout_embedding=0.2, # dropout for input embeddings (0: no dropout)
    dropout_hidden=0.2, # dropout for hidden layers (0: no dropout)
    dropout_source=0.1, # dropout source words (0: no dropout)
    dropout_target=0.1, # dropout target words (0: no dropout)
    model_version = 0.1
    ):
    
    #获取局部参数
    model_options = locals().copy()  
    print 'Model options:',model_options
    
    #加载字典，并且反转
    worddicts = [None]*len(dictionaries)
    worddicts_r = [None]*len(dictionaries)
    for ii,dd in enumerate(dictionaries):
        worddicts[ii] = load_dict(dd)
        worddicts_r[ii] = dict()
        for kk,vv in worddicts[ii].iteritems():
            worddicts_r[ii][vv] = kk
    
    #若词汇总大小未设置，则给定默认值为词汇表大小
    if n_words_src is None:
        n_words_src = len(worddicts[0])
        model_options['n_words_src'] = n_words_src
    if n_words_tgt is None:
        n_words_tgt = len(worddicts[1])
        model_options['n_words_tgt'] = n_words_tgt
    
    #加载数据
    print 'Loading data ...'
    train = TextIterator(datasets[0],datasets[1],hter[0],
                        dictionaries[0],dictionaries[1],
                        n_words_source=n_words_src,
                        n_words_target=n_words_tgt,
                        batch_size=batch_size,
                        maxlen=maxlen,
                        shuffle_each_epoch=shuffle_each_epoch,
                        sort_by_length=sort_by_length,
                        maxibatch_size=maxibatch_size)
    valid = TextIterator(valid_datasets[0], valid_datasets[1],hter[1],
                        dictionaries[0], dictionaries[1],
                        n_words_source=n_words_src, n_words_target=n_words_tgt,
                        batch_size=valid_batch_size,
                        maxlen=maxlen)
    
    numpy.random.seed(1234)
    # 初始化模型参数
    print 'Init parameters ...'
    params = init_params(model_options)
    
    #reload parameters
    if reload_ and os.path.exists(saveto):
        print 'Reloading model parameters'
        params = load_params(saveto,params)

    #把网络中的W，b 变为共享变量
    tparams = init_theano_params(params)
   
    # 建立模型
    print 'Building model ...',
    trng,use_noise,x,x_mask,y,y_mask,hter, \
        y_pred,cost = build_model(tparams,model_options)
    print 'Done'
    
    ###################################################################
    """
    @function:调试
    f_old_cost = theano.function([x,x_mask,y,y_mask],old_cost)
    f_cost = theano.function([x,x_mask,y,y_mask,hter],cost)
    for x, y, hter in train:
            # 准备数据用于训练
            x, x_mask, y, y_mask = nmt.prepare_data(x, y, maxlen=maxlen,
                                                n_words_src=n_words_src,
                                                n_words=n_words_tgt)
            print f_old_cost(x,x_mask,y,y_mask)
            print f_cost(x,x_mask,y,y_mask)
            
    sys.exit(0)
    """
    ####################################################################
    """
    @function:调试
    print Wt.get_value().shape
    print tparams['W'].get_value().shape
     
    f_tt = theano.function([x,x_mask,y,y_mask],tt)
    f_emb = theano.function([x,x_mask,y,y_mask],emb)
    f_pred = theano.function([x,x_mask,y,y_mask],y_pred)
    f_cost = theano.function([x,x_mask,y,y_mask,hter],cost)
    
    
    for x, y, hter in train:
            # 准备数据用于训练
            x, x_mask, y, y_mask = nmt.prepare_data(x, y, maxlen=maxlen,
                                                n_words_src=n_words_src,
                                                n_words=n_words_tgt)
            hter = numpy.array(hter).astype('float32')
            hter = hter.reshape([hter.shape[0],1])

            print f_pred(x,x_mask,y,y_mask).shape
            print f_cost(x,x_mask,y,y_mask,hter)
            #print f_cost(x,x_mask,y,y_mask,hter)
            
            sys.exit(0)
    """
    ####################################################################
    
    # apply L2 regularization on weights
    if decay_c > 0.:
        decay_c = theano.shared(numpy.float32(decay_c), name='decay_c')
        weight_decay = 0.
        for kk, vv in tparams.iteritems():
            weight_decay += (vv ** 2).sum()
        weight_decay *= decay_c
        cost += weight_decay

     # apply L2 regularisation to loaded model (map training)
    if map_decay_c > 0:
        map_decay_c = theano.shared(numpy.float32(map_decay_c), name="map_decay_c")
        weight_map_decay = 0.
        for kk, vv in tparams.iteritems():
            init_value = theano.shared(vv.get_value(), name= kk + "_init")
            weight_map_decay += ((vv -init_value) ** 2).sum()
        weight_map_decay *= map_decay_c
        cost += weight_map_decay
    
    print 'Building f_pred...',
    inps = [x,x_mask,y,y_mask]
    f_pred = theano.function(inps,y_pred,profile=False)
    print 'Done'
    
    print 'Building f_cost...',
    inps = [x,x_mask,y,y_mask,hter]
    f_cost = theano.function(inps,cost,profile=False)
    
    print 'Done'

    print 'Computing gradient...',
    grads = tensor.grad(cost, wrt=itemlist(tparams))
    print 'Done'
    
    # apply gradient clipping here
    if clip_c > 0.:
        g2 = 0.
        for g in grads:
            g2 += (g**2).sum()
        new_grads = []
        for g in grads:
            new_grads.append(tensor.switch(g2 > (clip_c**2),
                                           g / tensor.sqrt(g2) * clip_c,
                                           g))
        grads = new_grads
        
    # compile the optimizer, the actual computational graph is compiled here
    lr = tensor.scalar(name='lr')
    print 'Building optimizers...',
    f_grad_shared, f_update = eval(optimizer)(lr, tparams, grads, inps, cost, profile=False)
    print 'Done'
    
    print 'Start Optimization'
    best_p = None
    bad_counter = 0
    uidx = 0
    estop = False
    history_errs = []
    # reload history
    if reload_ and os.path.exists(saveto):
        rmodel = numpy.load(saveto)
        history_errs = list(rmodel['history_errs'])
        if 'uidx' in rmodel:
            uidx = rmodel['uidx']

    if validFreq == -1:
        validFreq = len(train[0])/batch_size
    if saveFreq == -1:
        saveFreq = len(train[0])/batch_size

    valid_err = None
    fp = open('stack_model/valid.error','w')
    
    for eidx in xrange(max_epochs):
        
        n_samples = 0
        for x, y, hter in train:
            n_samples += len(x)
            uidx += 1
            use_noise.set_value(1.)
            # 准备数据用于训练
            x, x_mask, y, y_mask = nmt.prepare_data(x, y, maxlen=maxlen,
                                                n_words_src=n_words_src,
                                                n_words=n_words_tgt)
            hter = numpy.array(hter).astype('float32')
            hter = hter.reshape([hter.shape[0],1])
            
            #长度小于 maxlen 的值的句子为 0
            if x is None:
                print 'Minibatch with zero sample under length ', maxlen
                uidx -= 1
                continue

            ud_start = time.time()

            # compute cost, grads and copy grads to shared variables
            cost = f_grad_shared(x, x_mask, y, y_mask, hter)

            # do the update on parameters
            f_update(lrate)

            ud = time.time() - ud_start

            # check for bad numbers, usually we remove non-finite elements
            # and continue training - but not done here
            if numpy.isnan(cost) or numpy.isinf(cost):
                print 'NaN detected'
                return 1., 1., 1.

            # verbose
            if numpy.mod(uidx, dispFreq) == 0:
                print 'Epoch ', eidx, 'Update ', uidx, 'Cost ', cost, 'UD ', ud

            # save the best model so far, in addition, save the latest model
            # into a separate file with the iteration number for external eval
            if numpy.mod(uidx, saveFreq) == 0:
                print 'Saving the best model...',
                if best_p is not None:
                    params = best_p
                else:
                    params = unzip_from_theano(tparams)
                numpy.savez(saveto, history_errs=history_errs, uidx=uidx, **params)
                json.dump(model_options, open('%s.json' % saveto, 'wb'), indent=2)
                print 'Done'

                # save with uidx
                if not overwrite:
                    print 'Saving the model at iteration {}...'.format(uidx),
                    saveto_uidx = '{}.iter{}.npz'.format(
                        os.path.splitext(saveto)[0], uidx)
                    numpy.savez(saveto_uidx, history_errs=history_errs,
                                uidx=uidx, **unzip_from_theano(tparams))
                    print 'Done'

            # validate model on validation set and early stop if necessary
            if valid and validFreq and numpy.mod(uidx, validFreq) == 0:
                use_noise.set_value(0.)
                valid_errs = pred_error(f_cost, nmt.prepare_data,
                                        model_options, valid)
                valid_err = valid_errs.mean()
                history_errs.append(valid_err)

                if uidx == 0 or valid_err <= numpy.array(history_errs).min():
                    best_p = unzip_from_theano(tparams)
                    bad_counter = 0
                if len(history_errs) > patience and valid_err >= \
                        numpy.array(history_errs)[:-patience].min():
                    bad_counter += 1
                    if bad_counter > patience:
                            print 'Early Stop!'
                            estop = True
                            break

                if numpy.isnan(valid_err):
                    ipdb.set_trace()

                print 'Valid ', valid_err
                fp.writelines('valid error: '+str(valid_err)+'\n')
                    

            # finish after this many updates
            if uidx >= finish_after:
                print 'Finishing after %d iterations!' % uidx
                estop = True
                break

        print 'Seen %d samples' % n_samples

        if estop:
            break
    
    if best_p is not None:
        zip_to_theano(best_p, tparams)

    if valid:
        use_noise.set_value(0.)
        valid_errs = pred_error(f_cost, nmt.prepare_data,
                               model_options, valid)
        valid_err = valid_errs.mean()
        print 'Valid ', valid_err
        
        fp.writelines('Finally error: '+str(valid_err)+'\n')
        fp.close()
    
    if best_p is not None:
        params = copy.copy(best_p)
    else:
        params = unzip_from_theano(tparams)
    numpy.savez(saveto, zipped_params=best_p,
                history_errs=history_errs,
                uidx=uidx,
                **params)

    return valid_err

def init_params(options):
    params = OrderedDict()
    
    #获取nmt的参数
    nmt_options = load_config(options['nmt_model'])
    nmt_params = nmt.init_params(nmt_options)
    nmt_params = load_params(options['nmt_model'], nmt_params)
    for  key in nmt_params:
        params[key] = nmt_params[key]
    
    #获取final_rnn的参数
    final_options = load_config(options['final_model'])
    final_params = rnn.init_params(final_options)
    final_params = load_params(options['final_model'], final_params)
    for key in final_params:
        params[key] = final_params[key]
    
    return params
    
def build_model(tparams,options):    
    """
    @first:得到f_tt函数
    """
    nmt_options = load_config(options['nmt_model'])
    trng,use_noise,x,x_mask,y,y_mask,\
        opt_ret, nmt_cost, ctx, tt = nmt.build_model(tparams,nmt_options) # *** tparams ***
    
    #删除网络参数ff_logit_b，因为计算cost用不到此参数
    tparams.pop('ff_logit_b')
    
    hter = tensor.matrix('hter',dtype='float32')
    Wt = tparams['ff_logit_W']  #old_tparams['ff_logit_W']              
    
    n_timesteps = y.shape[0]
    n_samples = y.shape[1]
    
    emb = Wt.T[y.flatten()]
    emb = emb.reshape([n_timesteps,n_samples,options['dim_word']])
    emb = emb*tt
    
    #是否使用 dropout
    if options['use_dropout']:
        retain_probability_emb = 1-options['dropout_embedding']
        retain_probability_hidden = 1-options['dropout_hidden']
        retain_probability_source = 1-options['dropout_source']
        if options['model_version'] < 0.1:
            scaled = False
        else:
            scaled = True
        rec_dropout = shared_dropout_layer((2, n_samples, options['dim']), use_noise, trng, retain_probability_hidden, scaled)
        emb_dropout = shared_dropout_layer((2, n_samples, options['dim_word']), use_noise, trng, retain_probability_emb, scaled)
        source_dropout = shared_dropout_layer((n_timesteps, n_samples, 1), use_noise, trng, retain_probability_source, scaled)
        source_dropout = tensor.tile(source_dropout, (1,1,options['dim_word']))
    else:
        rec_dropout = theano.shared(numpy.array([1.]*2, dtype='float32'))
        emb_dropout = theano.shared(numpy.array([1.]*2, dtype='float32'))
    
    proj = gru_layer(tparams,emb,options,
                     prefix='final_encoder',
                     mask=y_mask,
                     emb_dropout=emb_dropout,
                     rec_dropout=rec_dropout,
                     profile=False)
    
    hh = proj[0][-1,:,:]
    y_pred = tensor.dot(hh,tparams['final_W'])  #此时得出的结果也不错

    final_cost = tensor.abs_(y_pred-hter).mean(axis=0)[0]
    cost = final_cost
    
    return trng,use_noise,x,x_mask,y,y_mask,hter,y_pred,cost

def pred_error(f_cost, prepare_data, options, valid):
    
    error = []
    for x,y,hter in valid:
        x, x_mask, y, y_mask = nmt.prepare_data(x, y, maxlen=options['maxlen'],
                                                n_words_src=options['n_words_src'],
                                                n_words=options['n_words_tgt'])
        hter = numpy.array(hter).astype('float32')
        hter = hter.reshape([hter.shape[0],1])
        error.append(f_cost(x,x_mask,y,y_mask,hter))
        
    error = numpy.array(error)
    
    return error
        


if __name__ == '__main__':
    pass