#!/usr/bin/python
#-*- coding:utf8 -*-

'''
Created on Apr 27, 2017

@author: czm
'''
import json
import numpy
import sys

from collections import OrderedDict

def get_word_freq(filename):
    word_freqs = OrderedDict()
    with open(filename) as fp:
            for lines in fp:
                words_in = lines.strip().split(' ')
                for w in words_in:
                    if w not in word_freqs:
                        word_freqs[w] = 0
                    word_freqs[w] += 1
    return word_freqs
    
def main():
    for filename in sys.argv[1:]:
        print 'Processing',filename
        #获得词频
        word_freqs = get_word_freq(filename)
        words = word_freqs.keys()
        freqs = word_freqs.values()
        
        sorted_idx = numpy.argsort(freqs) #返回从小到大的序号
        sorted_words = [words[ii] for ii in sorted_idx[::-1]]
        
        worddict = OrderedDict()
        #加入两个特殊字符
        worddict['eos'] = 0
        worddict['UNK'] = 1
        for ii,ww in enumerate(sorted_words):
            worddict[ww] = ii+2
        
        with open('%s.json'%filename,'wb') as fp:
            #保存的内容缩进两个字符
            json.dump(worddict,fp,indent=2,ensure_ascii=False)
        
        print 'Done'
        
        

if __name__ == '__main__':
    main()
