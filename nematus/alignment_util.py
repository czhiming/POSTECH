#-*- coding:utf8 -*-
'''
Created on May 3, 2017

@author: czm
'''
import json
import sys
import codecs

def get_alignments(attention, x_mask, y_mask):
    #print "\nPrinting Attention..."
    #print attention
    #print "\nPrinting x_mask, need to figure out how to use it"
    #print x_mask
    #print "\nPrinting y_mask, need to figure out how to use it"
    #print y_mask

    n_rows, n_cols = y_mask.shape  ###n_cols correspond to the number of sentences.
    #print "Number of rows and number of columns: \n\n", n_rows, n_cols

    for target_sent_index in range(n_cols):
        #print "\n\n","*" * 40
        print "Going through sentence", target_sent_index
        #source_sent_index = source_indexes[target_sent_index]
        target_length = y_mask[:,target_sent_index].tolist().count(1)
        source_length = x_mask[:,target_sent_index].tolist().count(1)
        # #print "STEP1: The attention matrix that is relevant for this sentence",
        temp_attention = attention[range(target_length),:,:]
        #print "STEP2: The attention matrix that is particular to just this sentence\n",
        this_attention = temp_attention[:,target_sent_index,range(source_length)]

        jdata = {}
        jdata['matrix'] = this_attention.tolist()
        jdata = json.dumps(jdata)
        #print "\t\tJSON Data"
        #print "\t\t",jdata
        yield jdata



















if __name__ == '__main__':
    pass