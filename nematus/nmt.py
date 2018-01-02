#-*- coding:utf8 -*-
'''
Created on Apr 27, 2017
@function: 基于注意力机制的编码器解码器框架
@author: czm
'''
from util import load_dict
from data_iterator import TextIterator
from initializers import norm_weight,ortho_weight
from layers import param_init_gru,param_init_fflayer,param_init_gru_cond,shared_dropout_layer,\
                    gru_layer,fflayer,gru_cond_layer
from theano_util import load_params,init_theano_params,concatenate,itemlist,unzip_from_theano,\
                zip_to_theano
from optimizers import *
from alignment_util import get_alignments

import numpy 
import theano
from theano import tensor
import sys
import os
from collections import OrderedDict
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
from click.decorators import option
import time
import copy
import ipdb
from subprocess import Popen
import matplotlib.pyplot as plt

profile = False


def train(
    dim_word=100,  # word vector dimensionality
    dim=1000,  # the number of LSTM units
    patience=10,  # early stopping patience
    max_epochs=5000,
    finish_after=10000000,  # finish after this many updates
    dispFreq=100,
    decay_c=0.,  # L2 regularization penalty
    map_decay_c=0., # L2 regularization penalty towards original weights
    alpha_c=0.,  # alignment regularization
    clip_c=-1.,  # gradient clipping threshold
    lrate=0.01,  # learning rate
    n_words_src=None,  # source vocabulary size
    n_words_tgt=None,  # target vocabulary size
    maxlen=100,  # maximum length of the description
    optimizer='rmsprop',
    batch_size=16,
    valid_batch_size=16,
    saveto='model.npz',
    validFreq=1000,
    saveFreq=1000,   # save the parameters after every saveFreq updates
    sampleFreq=100,   # generate some samples after every sampleFreq
    datasets=[
        '/data/lisatmp3/chokyun/europarl/europarl-v7.fr-en.en.tok',
        '/data/lisatmp3/chokyun/europarl/europarl-v7.fr-en.fr.tok'],
    valid_datasets=['../data/dev/newstest2011.en.tok',
                    '../data/dev/newstest2011.fr.tok'],
    dictionaries=[
        '/data/lisatmp3/chokyun/europarl/europarl-v7.fr-en.en.tok.pkl',
        '/data/lisatmp3/chokyun/europarl/europarl-v7.fr-en.fr.tok.pkl'],
    use_dropout=False,
    dropout_embedding=0.2, # dropout for input embeddings (0: no dropout)
    dropout_hidden=0.5, # dropout for hidden layers (0: no dropout)
    dropout_source=0, # dropout source words (0: no dropout)
    dropout_target=0, # dropout target words (0: no dropout)
    reload_=False,
    overwrite=False,
    external_validation_script=None,
    shuffle_each_epoch=True,
    sort_by_length=True,
    maxibatch_size=20, #How many minibatches to load at one time
    model_version = 0.1
    ):
    # 获取局部参数
    model_options = locals().copy()  
    print 'Model options:',model_options
    
    # 加载字典，并且反转
    worddicts = [None]*len(dictionaries)
    worddicts_r = [None]*len(dictionaries)
    for ii,dd in enumerate(dictionaries):
        worddicts[ii] = load_dict(dd)
        worddicts_r[ii] = dict()
        for kk,vv in worddicts[ii].iteritems():
            worddicts_r[ii][vv] = kk
    
    # 若词汇总大小未设置，则给定默认值为词汇表大小
    if n_words_src is None:
        n_words_src = len(worddicts[0])
        model_options['n_words_src'] = n_words_src
    if n_words_tgt is None:
        n_words_tgt = len(worddicts[1])
        model_options['n_words_tgt'] = n_words_tgt
    
    # 加载数据
    print 'Loading data ...'
    train = TextIterator(datasets[0],datasets[1],
                        dictionaries[0],dictionaries[1],
                        n_words_source=n_words_src,
                        n_words_target=n_words_tgt,
                        batch_size=batch_size,
                        maxlen=maxlen,
                        shuffle_each_epoch=shuffle_each_epoch,
                        sort_by_length=sort_by_length,
                        maxibatch_size=maxibatch_size)
    valid = TextIterator(valid_datasets[0], valid_datasets[1],
                        dictionaries[0], dictionaries[1],
                        n_words_source=n_words_src, n_words_target=n_words_tgt,
                        batch_size=valid_batch_size,
                        maxlen=maxlen)
    
    # 初始化模型参数
    print 'Init parameters ...'
    params = init_params(model_options)
    
    # 重新载入模型，当程序意外中断的时候，可以继续运行代码
    if reload_ and os.path.exists(saveto):
        print 'Reloading model parameters'
        params = load_params(saveto,params)
    
    # 把网络中的W，b 变为共享变量
    tparams = init_theano_params(params)
    
    # 建立模型
    print 'Building model ...'
    
    trng,use_noise,x,x_mask,y,y_mask,\
        opt_ret, cost, ctx, tt = build_model(tparams,model_options)
    
    inps = [x, x_mask, y, y_mask]

    #建立采样器
    if validFreq or sampleFreq:
        print 'Building sampler ...'
        f_init, f_next = build_sampler(tparams, model_options, use_noise, trng)

    # before any regularizer
    print 'Building f_log_probs...',
    f_log_probs = theano.function(inps, cost, profile=profile)
    print 'Done'

    cost = cost.mean()

    # apply L2 regularization on weights
    if decay_c > 0.:
        decay_c = theano.shared(numpy.float32(decay_c), name='decay_c')
        weight_decay = 0.
        for kk, vv in tparams.iteritems():
            weight_decay += (vv ** 2).sum()
        weight_decay *= decay_c
        cost += weight_decay

    # regularize the alpha weights
    if alpha_c > 0. and not model_options['decoder'].endswith('simple'):
        alpha_c = theano.shared(numpy.float32(alpha_c), name='alpha_c')
        alpha_reg = alpha_c * (
            (tensor.cast(y_mask.sum(0)//x_mask.sum(0), 'float32')[:, None] -
             opt_ret['dec_alphas'].sum(0))**2).sum(1).mean()
        cost += alpha_reg

     # apply L2 regularisation to loaded model (map training)
    if map_decay_c > 0:
        map_decay_c = theano.shared(numpy.float32(map_decay_c), name="map_decay_c")
        weight_map_decay = 0.
        for kk, vv in tparams.iteritems():
            init_value = theano.shared(vv.get_value(), name= kk + "_init")
            weight_map_decay += ((vv -init_value) ** 2).sum()
        weight_map_decay *= map_decay_c
        cost += weight_map_decay
    
    # after all regularizers - compile the computational graph for cost
    print 'Building f_cost...',
    f_cost = theano.function(inps, cost, profile=profile)
    f_alpha = theano.function(inps, opt_ret['dec_alphas']) # alphas
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
    f_grad_shared, f_update = eval(optimizer)(lr, tparams, grads, inps, cost, profile=profile)
    print 'Done'
    
    #开始优化
    print 'Optimization'

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
    if sampleFreq == -1:
        sampleFreq = len(train[0])/batch_size

    valid_err = None

    for eidx in xrange(max_epochs):
        n_samples = 0

        for x, y in train:
            n_samples += len(x)
            uidx += 1
            use_noise.set_value(1.)
            # 准备数据用于训练
            x, x_mask, y, y_mask = prepare_data(x, y, maxlen=maxlen,
                                                n_words_src=n_words_src,
                                                n_words=n_words_tgt)
            #长度小于 maxlen 的值的句子为 0
            if x is None:
                print 'Minibatch with zero sample under length ', maxlen
                uidx -= 1
                continue

            ud_start = time.time()

            # compute cost, grads and copy grads to shared variables
            cost = f_grad_shared(x, x_mask, y, y_mask)
            
            # 画出词对齐矩阵
            #print f_alpha(x, x_mask, y, y_mask).shape
            """
            x_word = [worddicts_r[0][idx] for idx in x[:,0]]
            y_word = [worddicts_r[1][idx] for idx in y[:,0]]
            print len(x_word), x_word
            print len(y_word), y_word
            shape = f_alpha(x, x_mask, y, y_mask).shape
            for i in range(shape[1]):
                # print sum(f_alpha(x, x_mask, y, y_mask)[i,0,:])
                mx = sum(y_mask[:,i])
                my = sum(x_mask[:,i])
                align_matrix = f_alpha(x, x_mask, y, y_mask)[:,i,:][0:mx,0:my]
                align_shape = align_matrix.shape
                scale_ = 20 # 图像大小
                out_matrix = numpy.ones([scale_*align_shape[0],scale_*align_shape[1]])
                for j in range(align_shape[0]):
                    for k in range(align_shape[1]):
                        out_matrix[j*scale_:(j+1)*scale_,k*scale_:(k+1)*scale_] *= align_matrix[j,k]
                
                plt.imshow(100*out_matrix, plt.cm.gray)
                plt.pause(1)
                
            plt.show()
            sys.exit(0)
            """
            
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


            # generate some samples with the model and display them
            
            if sampleFreq and numpy.mod(uidx, sampleFreq) == 0:
                # FIXME: random selection?
                for jj in xrange(numpy.minimum(5, x.shape[1])):
                    stochastic = True
                    sample, score, sample_word_probs, alignment = gen_sample([f_init], [f_next],
                                               x[:, jj][:, None],
                                               trng=trng, k=1,
                                               maxlen=30,
                                               stochastic=stochastic,
                                               argmax=False,
                                               suppress_unk=False)
                    print 'Source ', jj, ': ',
                    for vv in x[:,jj]:
                        if vv == 0:
                            break
                        if vv in worddicts_r[0]:
                            print worddicts_r[0][vv],
                        else:
                            print 'UNK'
                    print
                    print 'Truth ', jj, ' : ',
                    for vv in y[:, jj]:
                        if vv == 0:
                            break
                        if vv in worddicts_r[-1]:
                            print worddicts_r[-1][vv],
                        else:
                            print 'UNK',
                    print
                    print 'Sample ', jj, ': ',
                    if stochastic:
                        ss = sample
                    else:
                        score = score / numpy.array([len(s) for s in sample])
                        ss = sample[score.argmin()]
                    for vv in ss:
                        if vv == 0:
                            break
                        if vv in worddicts_r[-1]:
                            print worddicts_r[-1][vv],
                        else:
                            print 'UNK',
                    print
            
            # validate model on validation set and early stop if necessary
            if valid and validFreq and numpy.mod(uidx, validFreq) == 0:
                use_noise.set_value(0.)
                valid_errs, alignment = pred_probs(f_log_probs, prepare_data,
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

                if external_validation_script:
                    print "Calling external validation script"
                    print 'Saving  model...',
                    params = unzip_from_theano(tparams)
                    #每次验证的时候，也会保存 uidx
                    numpy.savez(saveto +'.dev', history_errs=history_errs, uidx=uidx, **params)
                    json.dump(model_options, open('%s.dev.npz.json' % saveto, 'wb'), indent=2)
                    print 'Done'
                    p = Popen([external_validation_script])

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
        valid_errs, alignment = pred_probs(f_log_probs, prepare_data,
                                        model_options, valid)
        valid_err = valid_errs.mean()

        print 'Valid ', valid_err

    if best_p is not None:
        params = copy.copy(best_p)
    else:
        params = unzip_from_theano(tparams)
    numpy.savez(saveto, zipped_params=best_p,
                history_errs=history_errs,
                uidx=uidx,
                **params)

    return valid_err

#initialize all parameters
def init_params(options):
    params = OrderedDict()
    
    #embedding
    params['Wemb'] = norm_weight(options['n_words_src'],
                                 options['dim_word'])
    params['Wemb_dec'] = norm_weight(options['n_words_tgt'],
                                     options['dim_word'])
    #encoder: bidirectional RNN
    params = param_init_gru(options,params,
                            prefix='encoder',
                            nin=options['dim_word'],
                            dim=options['dim'])
    params = param_init_gru(options,params,
                           prefix='encoder_r',
                           nin=options['dim_word'],
                           dim=options['dim'])
    ctxdim = 2*options['dim']
    #init state, init cell
    params = param_init_fflayer(options,params,prefix='ff_state',
                                nin=ctxdim,nout=options['dim'])
    #decoder
    params = param_init_gru_cond(options,params,
                                 prefix='decoder',
                                 nin=options['dim_word'],
                                 dim=options['dim'],
                                 dimctx=ctxdim)
    #readout
    params = param_init_fflayer(options,params,prefix='ff_logit_lstm',
                                nin=options['dim'],nout=options['dim_word'],
                                ortho=False)
    params = param_init_fflayer(options,params,prefix='ff_logit_prev',
                                nin=options['dim_word'],
                                nout=options['dim_word'],ortho=False)
    params = param_init_fflayer(options,params,prefix='ff_logit_ctx',
                                nin=ctxdim,nout=options['dim_word'],
                                ortho=False)
    params = param_init_fflayer(options,params,prefix='ff_logit', 
                                 nin=options['dim_word'],
                                 nout=options['n_words_tgt'])
    
    return params
    
#build a training model
def build_model(tparams,options):
    """
    @function:建立模型
    """
    opt_ret = dict()
    
    trng = RandomStreams(1234)
    use_noise = theano.shared(numpy.float32(0.)) 
    
    x_mask = tensor.matrix('x_mask',dtype='float32')
    y = tensor.matrix('y',dtype='int64')
    y_mask = tensor.matrix('y_mask',dtype='float32')
    
    # 编码器
    x,ctx = build_encoder(tparams,options,trng,use_noise,x_mask,sampling=False)
    n_samples = x.shape[1]
    n_timesteps_trg = y.shape[0]
    
    if options['use_dropout']:
        retain_probability_emb = 1-options['dropout_embedding']
        retain_probability_hidden = 1-options['dropout_hidden']
        retain_probability_target = 1-options['dropout_target']
       
        if options['model_version'] < 0.1:
            scaled = False
        else:
            scaled = True
        rec_dropout_d = shared_dropout_layer((5, n_samples, options['dim']), use_noise, trng, retain_probability_hidden, scaled)
        emb_dropout_d = shared_dropout_layer((2, n_samples, options['dim_word']), use_noise, trng, retain_probability_emb, scaled)
        ctx_dropout_d = shared_dropout_layer((4, n_samples, 2*options['dim']), use_noise, trng, retain_probability_hidden, scaled)
        target_dropout = shared_dropout_layer((n_timesteps_trg, n_samples, 1), use_noise, trng, retain_probability_target, scaled)
        target_dropout = tensor.tile(target_dropout, (1,1,options['dim_word']))
    else:
        rec_dropout_d = theano.shared(numpy.array([1.]*5, dtype='float32'))
        emb_dropout_d = theano.shared(numpy.array([1.]*2, dtype='float32'))
        ctx_dropout_d = theano.shared(numpy.array([1.]*4, dtype='float32'))
    
    # mean of the context (across time) will be used to intialize decoder rnn
    ctx_mean = (ctx*x_mask[:,:,None]).sum(0) / x_mask.sum(0)[:,None]
    # or you can use the last state of forward+backward encoder rnns
    # ctx_mean = concatenate([proj[0][-1],projr[0][-1]],axis=proj[0].ndim-2)
    
    if options['use_dropout']:
        ctx_mean *= shared_dropout_layer((n_samples,2*options['dim']),use_noise,trng,retain_probability_hidden,scaled)
    
    # initial decoder state
    init_state = fflayer(tparams,ctx_mean,options,
                          prefix='ff_state',activ='tanh')
    
    # word embedding (target), we will shift the target sequence one time step
    # to the right. This is done because of the bi-gram connections in the
    # readout and decoder rnn. The first target will be all zeros and we will
    # not condition on the last output.
    emb = tparams['Wemb_dec'][y.flatten()]
    emb = emb.reshape([n_timesteps_trg, n_samples, options['dim_word']])

    emb_shifted = tensor.zeros_like(emb)
    emb_shifted = tensor.set_subtensor(emb_shifted[1:], emb[:-1])
    emb = emb_shifted

    if options['use_dropout']:
        emb *= target_dropout
    
    # decoder - pass through the decoder conditional gru with attention
    proj = gru_cond_layer(tparams, emb, options,
                                prefix='decoder',
                                mask=y_mask, context=ctx,
                                context_mask=x_mask,
                                one_step=False,
                                init_state=init_state,
                                emb_dropout=emb_dropout_d,
                                ctx_dropout=ctx_dropout_d,
                                rec_dropout=rec_dropout_d,
                                profile=profile)
    
    # hidden states of the decoder gru
    proj_h = proj[0]

    # weighted averages of context, generated by attention module
    ctxs = proj[1]
    
    if options['use_dropout']:
        proj_h *= shared_dropout_layer((n_samples, options['dim']), use_noise, trng, retain_probability_hidden, scaled)
        emb *= shared_dropout_layer((n_samples, options['dim_word']), use_noise, trng, retain_probability_emb, scaled)
        ctxs *= shared_dropout_layer((n_samples, 2*options['dim']), use_noise, trng, retain_probability_hidden, scaled)
    
    # weights (alignment matrix) #####LIUCAN: this is where the attention vector is.
    opt_ret['dec_alphas'] = proj[2]

    # compute word probabilities
    logit_lstm = fflayer(tparams, proj_h, options,
                                    prefix='ff_logit_lstm', activ='linear')
    logit_prev = fflayer(tparams, emb, options,
                                    prefix='ff_logit_prev', activ='linear')
    logit_ctx = fflayer(tparams, ctxs, options,
                                   prefix='ff_logit_ctx', activ='linear')
    logit = tensor.tanh(logit_lstm+logit_prev+logit_ctx)

    if options['use_dropout']:
        logit *= shared_dropout_layer((n_samples, options['dim_word']), use_noise, trng, retain_probability_hidden, scaled)
    
    # 生成tj，用于获取质量向量
    tt = logit
    
    
    logit = fflayer(tparams, logit, options,
                                   prefix='ff_logit', activ='linear')
    logit_shp = logit.shape
    probs = tensor.nnet.softmax(logit.reshape([logit_shp[0]*logit_shp[1],
                                               logit_shp[2]]))

    # cost
    y_flat = y.flatten()
    y_flat_idx = tensor.arange(y_flat.shape[0]) * options['n_words_tgt'] + y_flat
    cost = -tensor.log(probs.flatten()[y_flat_idx])
    cost = cost.reshape([y.shape[0], y.shape[1]])
    cost = (cost * y_mask).sum(0)

    return trng, use_noise, x, x_mask, y, y_mask, opt_ret, cost, ctx, tt
    
#bidirectional RNN encoder: take input x(optionally with mask), and produce sequence of context
def build_encoder(tparams,options,trng,use_noise,x_mask=None,sampling=False):
    
    x = tensor.matrix('x',dtype='int64')
    x.tag.test_value = (numpy.random.rand(5,10)*100).astype('int64')
    
    #for the backward rnn, we just need to invert x
    xr = x[::-1]   #此处有区别 xr = x[:,::-1]
    if x_mask is None:  #测试的时候
        xr_mask = None
    else:
        xr_mask = x_mask[::-1]
    
    #时间步数，和样本个数
    n_timesteps = x.shape[0]
    n_samples = x.shape[1]
    
    #是否使用 dropout
    if options['use_dropout']:
        retain_probability_emb = 1-options['dropout_embedding']
        retain_probability_hidden = 1-options['dropout_hidden']
        retain_probability_source = 1-options['dropout_source']
        if sampling:
            if options['model_version'] < 0.1:
                rec_dropout = theano.shared(numpy.array([retain_probability_hidden]*2, dtype='float32'))
                rec_dropout_r = theano.shared(numpy.array([retain_probability_hidden]*2, dtype='float32'))
                emb_dropout = theano.shared(numpy.array([retain_probability_emb]*2, dtype='float32'))
                emb_dropout_r = theano.shared(numpy.array([retain_probability_emb]*2, dtype='float32'))
                source_dropout = theano.shared(numpy.float32(retain_probability_source))
            else:
                rec_dropout = theano.shared(numpy.array([1.]*2, dtype='float32'))
                rec_dropout_r = theano.shared(numpy.array([1.]*2, dtype='float32'))
                emb_dropout = theano.shared(numpy.array([1.]*2, dtype='float32'))
                emb_dropout_r = theano.shared(numpy.array([1.]*2, dtype='float32'))
                source_dropout = theano.shared(numpy.float32(1.))
        else:
            if options['model_version'] < 0.1:
                scaled = False
            else:
                scaled = True
            rec_dropout = shared_dropout_layer((2, n_samples, options['dim']), use_noise, trng, retain_probability_hidden, scaled)
            rec_dropout_r = shared_dropout_layer((2, n_samples, options['dim']), use_noise, trng, retain_probability_hidden, scaled)
            emb_dropout = shared_dropout_layer((2, n_samples, options['dim_word']), use_noise, trng, retain_probability_emb, scaled)
            emb_dropout_r = shared_dropout_layer((2, n_samples, options['dim_word']), use_noise, trng, retain_probability_emb, scaled)
            source_dropout = shared_dropout_layer((n_timesteps, n_samples, 1), use_noise, trng, retain_probability_source, scaled)
            source_dropout = tensor.tile(source_dropout, (1,1,options['dim_word']))
    else:
        rec_dropout = theano.shared(numpy.array([1.]*2, dtype='float32'))
        rec_dropout_r = theano.shared(numpy.array([1.]*2, dtype='float32'))
        emb_dropout = theano.shared(numpy.array([1.]*2, dtype='float32'))
        emb_dropout_r = theano.shared(numpy.array([1.]*2, dtype='float32'))
    
    # word embedding for forward rnn (source)
    emb = tparams['Wemb'][x.flatten()]     #此处不同
    emb = emb.reshape([n_timesteps,n_samples,options['dim_word']])
    if options['use_dropout']:
        emb *= source_dropout
    
    proj = gru_layer(tparams,emb,options,
                     prefix='encoder',
                     mask=x_mask,
                     emb_dropout=emb_dropout,
                     rec_dropout=rec_dropout,
                     profile=profile)
    
    # word embedding for backward rnn (source)
    embr = tparams['Wemb'][xr.flatten()]
    embr = embr.reshape([n_timesteps,n_samples,options['dim_word']])
    if options['use_dropout']:
        if sampling:
            embr *= source_dropout
        else:
            embr *= source_dropout[::-1]
    
    projr = gru_layer(tparams,embr,options,
                      prefix='encoder_r',
                      mask=xr_mask,
                      emb_dropout=emb_dropout_r,
                      rec_dropout=rec_dropout,
                      profile=profile)
    
    #context will be the concatenation of forward and backward rnns
    ctx = concatenate([proj[0],projr[0][::-1]],axis=proj[0].ndim-1)
    
    return x,ctx

# build a sampler
def build_sampler(tparams, options, use_noise, trng, return_alignment=False):

    if options['use_dropout'] and options['model_version'] < 0.1:
        retain_probability_emb = 1-options['dropout_embedding']
        retain_probability_hidden = 1-options['dropout_hidden']
        retain_probability_source = 1-options['dropout_source']
        retain_probability_target = 1-options['dropout_target']
        rec_dropout_d = theano.shared(numpy.array([retain_probability_hidden]*5, dtype='float32'))
        emb_dropout_d = theano.shared(numpy.array([retain_probability_emb]*2, dtype='float32'))
        ctx_dropout_d = theano.shared(numpy.array([retain_probability_hidden]*4, dtype='float32'))
        target_dropout = theano.shared(numpy.float32(retain_probability_target))
    else:
        rec_dropout_d = theano.shared(numpy.array([1.]*5, dtype='float32'))
        emb_dropout_d = theano.shared(numpy.array([1.]*2, dtype='float32'))
        ctx_dropout_d = theano.shared(numpy.array([1.]*4, dtype='float32'))

    x, ctx = build_encoder(tparams, options, trng, use_noise, x_mask=None, sampling=True)
    n_samples = x.shape[0]

    # get the input for decoder rnn initializer mlp
    ctx_mean = ctx.mean(0)
    # ctx_mean = concatenate([proj[0][-1],projr[0][-1]], axis=proj[0].ndim-2)

    if options['use_dropout'] and options['model_version'] < 0.1:
        ctx_mean *= retain_probability_hidden

    init_state = fflayer(tparams, ctx_mean, options,
                                    prefix='ff_state', activ='tanh')

    print >>sys.stderr, 'Building f_init...',
    outs = [init_state, ctx]
    f_init = theano.function([x], outs, name='f_init', profile=profile)
    print >>sys.stderr, 'Done'

    # x: 1 x 1
    y = tensor.vector('y_sampler', dtype='int64')
    init_state = tensor.matrix('init_state', dtype='float32')

    # if it's the first word, emb should be all zero and it is indicated by -1
    emb = tensor.switch(y[:, None] < 0,
                        tensor.alloc(0., 1, tparams['Wemb_dec'].shape[1]),
                        tparams['Wemb_dec'][y])

    if options['use_dropout'] and options['model_version'] < 0.1:
        emb *= target_dropout

    # apply one step of conditional gru with attention
    proj = gru_cond_layer(tparams, emb, options,
                                            prefix='decoder',
                                            mask=None, context=ctx,
                                            one_step=True,
                                            init_state=init_state,
                                            emb_dropout=emb_dropout_d,
                                            ctx_dropout=ctx_dropout_d,
                                            rec_dropout=rec_dropout_d,
                                            profile=profile)
    # get the next hidden state
    next_state = proj[0]

    # get the weighted averages of context for this target word y
    ctxs = proj[1]

    # alignment matrix (attention model)
    dec_alphas = proj[2]

    if options['use_dropout'] and options['model_version'] < 0.1:
        next_state_up = next_state * retain_probability_hidden
        emb *= retain_probability_emb
        ctxs *= retain_probability_hidden
    else:
        next_state_up = next_state

    logit_lstm = fflayer(tparams, next_state_up, options,
                                    prefix='ff_logit_lstm', activ='linear')
    logit_prev = fflayer(tparams, emb, options,
                                    prefix='ff_logit_prev', activ='linear')
    logit_ctx = fflayer(tparams, ctxs, options,
                                   prefix='ff_logit_ctx', activ='linear')
    logit = tensor.tanh(logit_lstm+logit_prev+logit_ctx)

    if options['use_dropout'] and options['model_version'] < 0.1:
        logit *= retain_probability_hidden

    logit = fflayer(tparams, logit, options,
                              prefix='ff_logit', activ='linear')

    # compute the softmax probability
    next_probs = tensor.nnet.softmax(logit)

    # sample from softmax distribution to get the sample
    next_sample = trng.multinomial(pvals=next_probs).argmax(1)

    # compile a function to do the whole thing above, next word probability,
    # sampled word for the next target, next hidden state to be used
    print >>sys.stderr, 'Building f_next..',
    inps = [y, ctx, init_state]
    outs = [next_probs, next_sample, next_state]

    if return_alignment:
        outs.append(dec_alphas)

    f_next = theano.function(inps, outs, name='f_next', profile=profile)
    print >>sys.stderr, 'Done'

    return f_init, f_next

# generate sample, either with stochastic sampling or beam search. Note that,
# this function iteratively calls f_init and f_next functions.
def gen_sample(f_init, f_next, x, trng=None, k=1, maxlen=30,
               stochastic=True, argmax=False, return_alignment=False, suppress_unk=False):

    # k is the beam size we have
    if k > 1:
        assert not stochastic, \
            'Beam search does not support stochastic sampling'

    sample = []
    sample_score = []
    sample_word_probs = []
    alignment = []
    if stochastic:
        sample_score = 0

    live_k = 1
    dead_k = 0

    hyp_samples = [[]] * live_k
    word_probs = [[]] * live_k
    hyp_scores = numpy.zeros(live_k).astype('float32')
    hyp_states = []
    if return_alignment:
        hyp_alignment = [[] for _ in xrange(live_k)]

    # for ensemble decoding, we keep track of states and probability distribution
    # for each model in the ensemble
    num_models = len(f_init)
    next_state = [None]*num_models
    ctx0 = [None]*num_models
    next_p = [None]*num_models
    dec_alphas = [None]*num_models
    # get initial state of decoder rnn and encoder context
    for i in xrange(num_models):
        ret = f_init[i](x)
        next_state[i] = ret[0]
        ctx0[i] = ret[1]
    next_w = -1 * numpy.ones((1,)).astype('int64')  # bos indicator

    # x is a sequence of word ids followed by 0, eos id
    for ii in xrange(maxlen):
        for i in xrange(num_models):
            ctx = numpy.tile(ctx0[i], [live_k, 1])
            inps = [next_w, ctx, next_state[i]]
            ret = f_next[i](*inps)
            # dimension of dec_alpha (k-beam-size, number-of-input-hidden-units)
            next_p[i], next_w_tmp, next_state[i] = ret[0], ret[1], ret[2]
            if return_alignment:
                dec_alphas[i] = ret[3]

            if suppress_unk:
                next_p[i][:,1] = -numpy.inf
        if stochastic:
            if argmax:
                nw = sum(next_p)[0].argmax()
            else:
                nw = next_w_tmp[0]
            sample.append(nw)
            sample_score += numpy.log(next_p[0][0, nw])
            if nw == 0:
                break
        else:
            cand_scores = hyp_scores[:, None] - sum(numpy.log(next_p))
            probs = sum(next_p)/num_models
            cand_flat = cand_scores.flatten()
            probs_flat = probs.flatten()
            ranks_flat = cand_flat.argpartition(k-dead_k-1)[:(k-dead_k)]

            #averaging the attention weights accross models
            if return_alignment:
                mean_alignment = sum(dec_alphas)/num_models

            voc_size = next_p[0].shape[1]
            # index of each k-best hypothesis
            trans_indices = ranks_flat / voc_size
            word_indices = ranks_flat % voc_size
            costs = cand_flat[ranks_flat]

            new_hyp_samples = []
            new_hyp_scores = numpy.zeros(k-dead_k).astype('float32')
            new_word_probs = []
            new_hyp_states = []
            if return_alignment:
                # holds the history of attention weights for each time step for each of the surviving hypothesis
                # dimensions (live_k * target_words * source_hidden_units]
                # at each time step we append the attention weights corresponding to the current target word
                new_hyp_alignment = [[] for _ in xrange(k-dead_k)]

            # ti -> index of k-best hypothesis
            for idx, [ti, wi] in enumerate(zip(trans_indices, word_indices)):
                new_hyp_samples.append(hyp_samples[ti]+[wi])
                new_word_probs.append(word_probs[ti] + [probs_flat[ranks_flat[idx]].tolist()])
                new_hyp_scores[idx] = copy.copy(costs[idx])
                new_hyp_states.append([copy.copy(next_state[i][ti]) for i in xrange(num_models)])
                if return_alignment:
                    # get history of attention weights for the current hypothesis
                    new_hyp_alignment[idx] = copy.copy(hyp_alignment[ti])
                    # extend the history with current attention weights
                    new_hyp_alignment[idx].append(mean_alignment[ti])


            # check the finished samples
            new_live_k = 0
            hyp_samples = []
            hyp_scores = []
            hyp_states = []
            word_probs = []
            if return_alignment:
                hyp_alignment = []

            # sample and sample_score hold the k-best translations and their scores
            for idx in xrange(len(new_hyp_samples)):
                if new_hyp_samples[idx][-1] == 0:
                    sample.append(new_hyp_samples[idx])
                    sample_score.append(new_hyp_scores[idx])
                    sample_word_probs.append(new_word_probs[idx])
                    if return_alignment:
                        alignment.append(new_hyp_alignment[idx])
                    dead_k += 1
                else:
                    new_live_k += 1
                    hyp_samples.append(new_hyp_samples[idx])
                    hyp_scores.append(new_hyp_scores[idx])
                    hyp_states.append(new_hyp_states[idx])
                    word_probs.append(new_word_probs[idx])
                    if return_alignment:
                        hyp_alignment.append(new_hyp_alignment[idx])
            hyp_scores = numpy.array(hyp_scores)

            live_k = new_live_k

            if new_live_k < 1:
                break
            if dead_k >= k:
                break

            next_w = numpy.array([w[-1] for w in hyp_samples])
            next_state = [numpy.array(state) for state in zip(*hyp_states)]

    if not stochastic:
        # dump every remaining one
        if live_k > 0:
            for idx in xrange(live_k):
                sample.append(hyp_samples[idx])
                sample_score.append(hyp_scores[idx])
                sample_word_probs.append(word_probs[idx])
                if return_alignment:
                    alignment.append(hyp_alignment[idx])

    if not return_alignment:
        alignment = [None for i in range(len(sample))]

    return sample, sample_score, sample_word_probs, alignment



# batch preparation
def prepare_data(seqs_x, seqs_y, maxlen=None, n_words_src=30000,
                 n_words=30000):
    # x: a list of sentences
    lengths_x = [len(s) for s in seqs_x]
    lengths_y = [len(s) for s in seqs_y]

    if maxlen is not None:
        new_seqs_x = []
        new_seqs_y = []
        new_lengths_x = []
        new_lengths_y = []
        for l_x, s_x, l_y, s_y in zip(lengths_x, seqs_x, lengths_y, seqs_y):
            if l_x < maxlen and l_y < maxlen:
                new_seqs_x.append(s_x)
                new_lengths_x.append(l_x)
                new_seqs_y.append(s_y)
                new_lengths_y.append(l_y)
        lengths_x = new_lengths_x
        seqs_x = new_seqs_x
        lengths_y = new_lengths_y
        seqs_y = new_seqs_y
        #所有的句子都不满足长度要求
        if len(lengths_x) < 1 or len(lengths_y) < 1:
            return None, None, None, None

    n_samples = len(seqs_x)
    #n_factors = len(seqs_x[0][0])
    maxlen_x = numpy.max(lengths_x) + 1
    maxlen_y = numpy.max(lengths_y) + 1

    x = numpy.zeros((maxlen_x, n_samples)).astype('int64')
    y = numpy.zeros((maxlen_y, n_samples)).astype('int64')
    x_mask = numpy.zeros((maxlen_x, n_samples)).astype('float32')
    y_mask = numpy.zeros((maxlen_y, n_samples)).astype('float32')
    for idx, [s_x, s_y] in enumerate(zip(seqs_x, seqs_y)):
        x[:lengths_x[idx], idx] = s_x
        x_mask[:lengths_x[idx]+1, idx] = 1.  #为何要加1，<eos> 结尾必须加上
        y[:lengths_y[idx], idx] = s_y
        y_mask[:lengths_y[idx]+1, idx] = 1.
        

    return x, x_mask, y, y_mask
    
# calculate the log probablities on a given corpus using translation model
def pred_probs(f_log_probs, prepare_data, options, iterator, verbose=True, normalize=False, alignweights=False):
    probs = []
    n_done = 0

    alignments_json = []

    for x, y in iterator:

        n_done += len(x)

        x, x_mask, y, y_mask = prepare_data(x, y,
                                            n_words_src=options['n_words_src'],
                                            n_words=options['n_words_tgt'])

        ### in optional save weights mode.
        if alignweights:
            pprobs, attention = f_log_probs(x, x_mask, y, y_mask)
            for jdata in get_alignments(attention, x_mask, y_mask):
                alignments_json.append(jdata)
        else:
            pprobs = f_log_probs(x, x_mask, y, y_mask)

        # normalize scores according to output length
        if normalize:
            lengths = numpy.array([numpy.count_nonzero(s) for s in y_mask.T])
            pprobs /= lengths

        for pp in pprobs:
            probs.append(pp)

        if numpy.isnan(numpy.mean(probs)):
            ipdb.set_trace()

        if verbose:
            print >>sys.stderr, '%d samples computed' % (n_done)

    return numpy.array(probs), alignments_json
    

if __name__ == '__main__':
    pass