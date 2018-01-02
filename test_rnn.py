#-*- coding:utf8 -*-
'''
Created on Jun 18, 2017

@author: czm
'''
from qe import qe_rnn, qe_rnn_trg, qe_rnn_stack_pro


if __name__ == '__main__':
    
    qe_rnn_stack_pro(
        datasets=['test/test17.bpe.en',
                  'test/test17.bpe.de'],
        save_file='test17.hter.pred'
        )
    
    
    
    
    
    