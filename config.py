#-*- coding:utf8 -*-
'''
Created on Apr 27, 2017

@author: czm
'''
import numpy
import os
import sys

#此处根据需要修改，看词汇表的个数多少
VOCAB_SIZE_SRC = 40000
VOCAB_SIZE_TGT = 40000
SRCTAG = "en"
TRGTAG = "de"
DATA_DIR = "data"
TUNING_DIR = "tuning"

from nematus.nmt import train


if __name__ == '__main__':
    validerr = train(saveto='model/model.npz',
                    reload_=True,
                    dim_word=500,
                    dim=1024,
                    n_words_tgt=VOCAB_SIZE_TGT,
                    n_words_src=VOCAB_SIZE_SRC,
                    decay_c=0.,
                    clip_c=1.,
                    lrate=0.0001,
                    optimizer='adam', #adam,adadelta
                    maxlen=50,
                    batch_size=80,
                    valid_batch_size=80,
                    datasets=[DATA_DIR + '/train.bpe.' + SRCTAG, DATA_DIR + '/train.bpe.' + TRGTAG],
                    valid_datasets=[DATA_DIR + '/dev.bpe.' + SRCTAG, DATA_DIR + '/dev.bpe.' + TRGTAG],
                    dictionaries=[DATA_DIR + '/train.bpe.' + SRCTAG + '.json',DATA_DIR + '/train.bpe.' + TRGTAG + '.json'],
                    validFreq=10000, #10000,3000
                    dispFreq=1000,  #1000,100
                    saveFreq=30000, #30000,10000
                    #sampleFreq=10000,
                    sampleFreq=0,  #不产生样本
                    use_dropout=True,
                    dropout_embedding=0.2, # dropout for input embeddings (0: no dropout)
                    dropout_hidden=0.2, # dropout for hidden layers (0: no dropout)
                    dropout_source=0.1, # dropout source words (0: no dropout)
                    dropout_target=0.1, # dropout target words (0: no dropout)
                    overwrite=False,
                    external_validation_script='./validate.sh')
    print validerr
