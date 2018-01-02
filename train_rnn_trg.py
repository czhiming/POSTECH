#-*- coding:utf8 -*-
'''
Created on Jun 16, 2017

@author: czm
'''
from RNN.rnn_trg import train

if __name__ == '__main__':
    
    valid_errors =train(
                    batch_size=60,
                    valid_batch_size=60,
                    dim=100, #LSTM 隐单元维数
                    dim_word=500, # 词向量维数，和 NMT 模型相同
                    
                    dispFreq=1,
                    saveFreq=1000,
                    validFreq=100,
                    
                    saveto='RNN_trg_model/wmt17.en-de.npz',
                    datasets=['test/train.bpe.en',
                              'test/train.bpe.de'],
                    valid_datasets=['test/dev.bpe.en',
                                    'test/dev.bpe.de'],
                    dictionaries=['data/train.bpe.en.json',
                                  'data/train.bpe.de.json'], # 此处不同
                    hter=['test/train.hter',
                          'test/dev.hter'],
                    n_words_src=40000, # 和 NMT 模型相同
                    n_words_tgt=40000,
                    nmt_model='model/model.npz.best_bleu',
                    lrate=0.0001,  # learning rate
                    use_dropout=True,
                    patience=10,
                    optimizer='adadelta',
                    shuffle_each_epoch=True,
                    reload_=True,
                    overwrite=False,
                    sort_by_length=False,
                    maxlen=1000,
                    decay_c=0.,  # L2 regularization penalty
                    map_decay_c=0., # L2 regularization penalty towards original weights
                    clip_c=1.0
                )
    
    print valid_errors







