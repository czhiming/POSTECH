#-*- coding:utf8 -*-
'''
Created on Apr 30, 2017
@function: 数据处理
@author: czm
'''
import numpy
import gzip
from nematus import shuffle
from util import load_dict

def fopen(filename,mode='r'):
    if filename.endswith('.gz'):
        return gzip.open(filename,mode)
    return open(filename,mode)

class TextIterator:
    """
    @function:文档迭代器
    """
    def __init__(self,source,target,
                 source_dict,target_dict,
                 batch_size=128,
                 maxlen=100,
                 n_words_source=-1,
                 n_words_target=-1,
                 shuffle_each_epoch=False,
                 sort_by_length=True,
                 maxibatch_size=20):
        # 每次epoch都，打乱文件顺序
        if shuffle_each_epoch: 
            shuffle.main([source,target])
            self.source = fopen(source+'.shuf')
            self.target = fopen(target+'.shuf')
        else:
            self.source = fopen(source)
            self.target = fopen(target)
        
        self.source_dict = load_dict(source_dict)
        self.target_dict = load_dict(target_dict)
        self.batch_size = batch_size
        self.maxlen = maxlen
        self.n_words_source = n_words_source
        self.n_words_target = n_words_target
        
        if self.n_words_source > 0:
            for key,idx in self.source_dict.items():
                if idx >= self.n_words_source:
                    del self.source_dict[key]
        if self.n_words_target > 0:
            for key,idx in self.target_dict.items():
                if idx >= self.n_words_target:
                    del self.target_dict[key]
        
        self.shuffle = shuffle_each_epoch
        self.sort_by_length = sort_by_length
        self.source_buffer = []
        self.target_buffer = []
        self.k = batch_size*maxibatch_size
        self.end_of_data = False
    
    def __iter__(self):
        return self
    
    def reset(self):
        if self.shuffle:
            shuffle.main([self.source.name.replace('.shuf',''),
                          self.target.name.replace('.shuf','')])
            self.source = fopen(self.source.name)
            self.target = fopen(self.target.name)
        else:
            self.source.seek(0)
            self.target.seek(0)
    
    def next(self):
        
        if self.end_of_data:
            self.end_of_data = False
            self.reset()
            raise StopIteration
        
        source = []
        target = []
        
        #fill buffer, if it's empty
        assert len(self.source_buffer) == len(self.target_buffer)
        
        if len(self.source_buffer) == 0:
            for k_ in xrange(self.k):
                ss = self.source.readline()
                if ss == "":
                    break
                tt = self.target.readline()
                if tt == "":
                    break
                
                self.source_buffer.append(ss.strip().split())
                self.target_buffer.append(tt.strip().split())
            
            #sort by target buffer
            if self.sort_by_length:
                tlen = numpy.array([len(t) for t in self.target_buffer])
                tidx = tlen.argsort()
                
                _sbuf = [self.source_buffer[i] for i in tidx]
                _tbuf = [self.target_buffer[i] for i in tidx]
                
                self.source_buffer = _sbuf
                self.target_buffer = _tbuf
            else:
                self.source_buffer.reverse()
                self.target_buffer.reverse()
                
        if len(self.source_buffer) == 0 or len(self.target_buffer) == 0:
            self.end_of_data = False
            self.reset()
            raise StopIteration
        
        try:
            #actual work here
            while True:
                #read from source file and map to word index
                try:
                    ss = self.source_buffer.pop()
                except IndexError:
                    break
                ss = [self.source_dict[w] if w in self.source_dict else 1
                      for w in ss]
                if self.n_words_source > 0:
                    ss = [w if w < self.n_words_source else 1 for w in ss]
                
                tt = self.target_buffer.pop()
                tt = [self.target_dict[w] if w in self.target_dict else 1
                      for w in tt]
                if self.n_words_target > 0:
                    tt = [w if w < self.n_words_target else 1 for w in tt]
                
                if len(ss) > self.maxlen and len(tt) > self.maxlen:
                    continue
                if not ss or not tt:
                    continue
                
                source.append(ss)
                target.append(tt)
                
                if len(source) >= self.batch_size or \
                    len(target) >= self.batch_size:
                    break
        except IOError:
            self.end_of_data = True        
            
        #all sentence pairs in maxibatch filtered out because of length
        if len(source) == 0 or len(target) == 0:
            source,target = self.next()  
        
        
        return source,target


if __name__ == '__main__':
    pass