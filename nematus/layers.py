#-*- coding:utf8 -*-
'''
Created on Apr 30, 2017
@function: 各种层信息
@author: czm
'''
import numpy
from nematus.initializers import norm_weight,ortho_weight
from nematus.theano_util import pp,tanh,linear
import theano 
from theano import tensor
from setuptools.dist import sequence


#GRU layer
def param_init_gru(options,params,prefix='gru',nin=None,dim=None):
    """
    @function:初始化编码器GRU网络参数
    """
    if nin is None:
        nin = options['dim_proj']
    if dim is None:
        dim = options['dim_proj']
    
    #embedding to gates transformation weights, biases (Wz,Wr,bz,br)
    W = numpy.concatenate([norm_weight(nin,dim),
                           norm_weight(nin,dim)],axis=1)
    params[pp(prefix,'W')] = W
    params[pp(prefix,'b')] = numpy.zeros((2*dim,)).astype('float32')
    
    #recurrent transformation weights for gates (Uz,Ur)
    U = numpy.concatenate([ortho_weight(dim),
                           ortho_weight(dim)],axis=1)
    params[pp(prefix,'U')] = U
    
    #embedding to hidden state proposal weights, biases (Wt,Ut,bt)
    Wx = norm_weight(nin,dim)
    params[pp(prefix,'Wx')] = Wx
    params[pp(prefix,'bx')] = numpy.zeros((dim,)).astype('float32')
    
    #recurrent transformation weights for hidden state proposal
    Ux = ortho_weight(dim)
    params[pp(prefix,'Ux')] = Ux
    
    return params

def gru_layer(tparams,state_below,options,prefix='gru',mask=None,
              emb_dropout=None,
              rec_dropout=None,
              profile=False,
              **kwargs):
    """
    @function:编码器GRU层的计算
    """
    nsteps = state_below.shape[0]
    if state_below.ndim == 3:
        n_samples = state_below.shape[1]
    else:
        n_samples = 1
    
    dim = tparams[pp(prefix,'Ux')].shape[1]
    
    if mask is None:
        mask = tensor.alloc(1.,state_below.shape[0],1)
    
    # utility function to slice a tensor
    def _slice(_x,n,dim):
        if _x.ndim == 3:
            return _x[:,:,n*dim:(n+1)*dim]
        return _x[:,n*dim:(n+1)*dim]
    
    #state_below is the input word embeddings
    # input to the gates, concatenated
    state_below_ = tensor.dot(state_below*emb_dropout[0],tparams[pp(prefix,'W')]) + \
                tparams[pp(prefix,'b')]
    #input to compute the hidden state proposal
    state_belowx = tensor.dot(state_below*emb_dropout[1],tparams[pp(prefix,'Wx')]) + \
                tparams[pp(prefix,'bx')]
    
    # step function to be used by scan
    # arguments | sequences | outputs-info | non-seqs
    def _step_slice(m_,x_,xx_,h_,U,Ux,rec_dropout):
        
        preact = tensor.dot(h_*rec_dropout[0],U)
        preact += x_
        #reset and upate gates
        r = tensor.nnet.sigmoid(_slice(preact,0,dim))
        u = tensor.nnet.sigmoid(_slice(preact,1,dim))
        
        # compute the hidden state proposal
        preactx = tensor.dot(h_*rec_dropout[1],Ux)
        preactx = preactx*r
        preactx = preactx + xx_
        
        #hidden state proposal
        h = tensor.tanh(preactx)
        
        #leaky itegrate and obtain next hidden state
        h = u*h_ + (1.-u)*h
        h = m_[:,None]*h + (1.-m_)[:,None]*h_
        
        return h
    
    # prepare scan arugments
    seqs = [mask,state_below_,state_belowx]
    init_states = [tensor.alloc(0.,n_samples,dim)]
    _step = _step_slice
    shared_vars = [tparams[pp(prefix,'U')],
                   tparams[pp(prefix,'Ux')],
                   rec_dropout]
    
    rval,updates = theano.scan(_step,
                               sequences=seqs,
                               outputs_info=init_states,
                               non_sequences=shared_vars,
                               name=pp(prefix,'_layers'),
                               n_steps=nsteps,
                               profile=profile,
                               strict=True)
    rval = [rval]
    return rval

#feedforward layer
def param_init_fflayer(options,params,prefix='ff',nin=None,nout=None,
                       ortho=True):
    """
    @function:初始化前馈网络参数
    """
    if nin is None:
        nin = options['dim_proj']
    if nout is None:
        nout = options['dim_proj']
    params[pp(prefix,'W')] = norm_weight(nin,nout,scale=0.01,ortho=ortho)
    params[pp(prefix,'b')] = numpy.zeros((nout,)).astype('float32')
    
    return params

def fflayer(tparams,state_below,options,prefix='rconv',
            activ='lambda x:tensor.tanh(x)',**kwargs):
    """
    @function:前馈网络层
    """
    return eval(activ)(
        tensor.dot(state_below,tparams[pp(prefix,'W')])+
        tparams[pp(prefix,'b')])

#Conditional GRU layer with Attention
def param_init_gru_cond(options,params,prefix='gru_cond',
                        nin=None,dim=None,dimctx=None,
                        nin_nonlin=None,dim_nonlin=None):
    """
    @function:初始化解码器GRU网络参数
    """
    if nin is None:
        nin = options['dim']
    if dim is None:
        dim = options['dim']
    if dimctx is None:
        dimctx =options['dim']
    if nin_nonlin is None:
        nin_nonlin = nin
    if dim_nonlin is None:
        dim_nonlin = dim
    
    W = numpy.concatenate([norm_weight(nin,dim),
                           norm_weight(nin,dim)],axis=1)
    params[pp(prefix,'W')] = W
    params[pp(prefix,'b')] = numpy.zeros((2*dim,)).astype('float32')
    U = numpy.concatenate([ortho_weight(dim_nonlin),
                           ortho_weight(dim_nonlin)],axis=1)
    params[pp(prefix,'U')] = U
    
    Wx = norm_weight(nin_nonlin,dim_nonlin)
    params[pp(prefix,'Wx')] = Wx
    Ux = ortho_weight(dim_nonlin)
    params[pp(prefix,'Ux')] = Ux
    params[pp(prefix,'bx')] = numpy.zeros((dim_nonlin,)).astype('float32')
    
    U_n1 = numpy.concatenate([ortho_weight(dim_nonlin),
                              ortho_weight(dim_nonlin)],axis=1)
    params[pp(prefix,'U_n1')] = U_n1
    params[pp(prefix,'b_n1')] = numpy.zeros((2*dim_nonlin,)).astype('float32')
    Ux_n1 = ortho_weight(dim_nonlin)
    params[pp(prefix,'Ux_n1')] = Ux_n1
    params[pp(prefix,'bx_n1')] = numpy.zeros((dim_nonlin,)).astype('float32')
    
    #context to LSTM
    Wc = norm_weight(dimctx,dim*2)
    params[pp(prefix,'Wc')] = Wc
    Wcx = norm_weight(dimctx,dim)
    params[pp(prefix,'Wcx')] = Wcx
    
    #attention: context-> hidden
    W_comb_att = norm_weight(dim,dimctx)
    params[pp(prefix,'W_comb_att')] = W_comb_att
    
    #attention: context-> hidden
    Wc_att = norm_weight(dimctx)
    params[pp(prefix,'Wc_att')] = Wc_att
    
    #attention: hidden bias
    b_att = numpy.zeros((dimctx,)).astype('float32')
    params[pp(prefix,'b_att')] = b_att
    
    #attention:
    U_att = norm_weight(dimctx,1)
    params[pp(prefix,'U_att')] = U_att
    c_att = numpy.zeros((1,)).astype('float32')
    params[pp(prefix,'c_tt')] = c_att
    
    return params


def gru_cond_layer(tparams, state_below, options, prefix='gru',
                   mask=None, context=None, one_step=False,
                   init_memory=None, init_state=None,
                   context_mask=None, emb_dropout=None,
                   rec_dropout=None, ctx_dropout=None,
                   profile=False,
                   **kwargs):
    """
    @function:解码器GRU层的计算
    """
    assert context, 'Context must be provided'

    if one_step:
        assert init_state, 'previous state must be provided'

    nsteps = state_below.shape[0]
    if state_below.ndim == 3:
        n_samples = state_below.shape[1]
    else:
        n_samples = 1

    # mask
    if mask is None:
        mask = tensor.alloc(1., state_below.shape[0], 1)

    dim = tparams[pp(prefix, 'Wcx')].shape[1]

    # initial/previous state
    if init_state is None:
        init_state = tensor.alloc(0., n_samples, dim)

    # projected context
    assert context.ndim == 3, \
        'Context must be 3-d: #annotation x #sample x dim'
    pctx_ = tensor.dot(context*ctx_dropout[0], tparams[pp(prefix, 'Wc_att')]) +\
        tparams[pp(prefix, 'b_att')]

    def _slice(_x, n, dim):
        if _x.ndim == 3:
            return _x[:, :, n*dim:(n+1)*dim]
        return _x[:, n*dim:(n+1)*dim]

    # projected x
    state_belowx = tensor.dot(state_below*emb_dropout[0], tparams[pp(prefix, 'Wx')]) +\
        tparams[pp(prefix, 'bx')]
    state_below_ = tensor.dot(state_below*emb_dropout[1], tparams[pp(prefix, 'W')]) +\
        tparams[pp(prefix, 'b')]

    def _step_slice(m_, x_, xx_, h_, ctx_, alpha_, pctx_, cc_, rec_dropout, ctx_dropout,
                    U, Wc, W_comb_att, U_att, c_tt, Ux, Wcx,
                    U_n1, Ux_n1, b_n1, bx_n1):

        preact1 = tensor.dot(h_*rec_dropout[0], U)
        preact1 += x_
        preact1 = tensor.nnet.sigmoid(preact1)

        r1 = _slice(preact1, 0, dim)
        u1 = _slice(preact1, 1, dim)

        preactx1 = tensor.dot(h_*rec_dropout[1], Ux)
        preactx1 *= r1
        preactx1 += xx_

        h1 = tensor.tanh(preactx1)

        h1 = u1 * h_ + (1. - u1) * h1
        h1 = m_[:, None] * h1 + (1. - m_)[:, None] * h_

        # attention
        pstate_ = tensor.dot(h1*rec_dropout[2], W_comb_att)
        pctx__ = pctx_ + pstate_[None, :, :]
        #pctx__ += xc_
        pctx__ = tensor.tanh(pctx__)
        alpha = tensor.dot(pctx__*ctx_dropout[1], U_att)+c_tt
        alpha = alpha.reshape([alpha.shape[0], alpha.shape[1]])
        alpha = tensor.exp(alpha - alpha.max(0, keepdims=True))  #不同于 dl4mt
        if context_mask:
            alpha = alpha * context_mask
        alpha = alpha / alpha.sum(0, keepdims=True)
        ctx_ = (cc_ * alpha[:, :, None]).sum(0)  # current context

        preact2 = tensor.dot(h1*rec_dropout[3], U_n1)+b_n1
        preact2 += tensor.dot(ctx_*ctx_dropout[2], Wc)
        preact2 = tensor.nnet.sigmoid(preact2)

        r2 = _slice(preact2, 0, dim)
        u2 = _slice(preact2, 1, dim)

        preactx2 = tensor.dot(h1*rec_dropout[4], Ux_n1)+bx_n1
        preactx2 *= r2
        preactx2 += tensor.dot(ctx_*ctx_dropout[3], Wcx)

        h2 = tensor.tanh(preactx2)

        h2 = u2 * h1 + (1. - u2) * h2
        h2 = m_[:, None] * h2 + (1. - m_)[:, None] * h1

        return h2, ctx_, alpha.T  # pstate_, preact, preactx, r, u

    seqs = [mask, state_below_, state_belowx]
    #seqs = [mask, state_below_, state_belowx, state_belowc]
    _step = _step_slice

    shared_vars = [tparams[pp(prefix, 'U')],
                   tparams[pp(prefix, 'Wc')],
                   tparams[pp(prefix, 'W_comb_att')],
                   tparams[pp(prefix, 'U_att')],
                   tparams[pp(prefix, 'c_tt')],
                   tparams[pp(prefix, 'Ux')],
                   tparams[pp(prefix, 'Wcx')],
                   tparams[pp(prefix, 'U_n1')],
                   tparams[pp(prefix, 'Ux_n1')],
                   tparams[pp(prefix, 'b_n1')],
                   tparams[pp(prefix, 'bx_n1')]]

    if one_step:
        rval = _step(*(seqs + [init_state, None, None, pctx_, context, rec_dropout, ctx_dropout] +
                       shared_vars))
    else:
        rval, updates = theano.scan(_step,
                                    sequences=seqs,
                                    outputs_info=[init_state,
                                                  tensor.alloc(0., n_samples,
                                                               context.shape[2]),
                                                  tensor.alloc(0., n_samples,
                                                               context.shape[0])],
                                    non_sequences=[pctx_, context, rec_dropout, ctx_dropout]+shared_vars,
                                    name=pp(prefix, '_layers'),
                                    n_steps=nsteps,
                                    profile=profile,
                                    strict=True)
    return rval


# dropout that will be re-used at different time steps
def shared_dropout_layer(shape,use_noise,trng,value,scaled=True):
    """
    @function:dropout层
    """
    if scaled:
        proj = tensor.switch(
                use_noise,
                trng.binomial(shape,p=value,n=1,dtype='float32')/value,
                theano.shared(numpy.float32(1.))
                )
    else:
        proj = tensor.switch(
                use_noise,
                trng.binomial(shape,p=value,n=1,dt='float32'),
                theano.shared(numpy.float32(value))
                )
    return proj
    

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    


if __name__ == '__main__':
    pass