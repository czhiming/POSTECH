#-*- coding:utf8 -*-
'''
Created on Mar 2, 2017

@author: czm
@use:python make_csv.py test.hter.pred system_name
'''
import sys
import numpy

if len(sys.argv) < 3:
    sys.exit(0)
    
hter_file = sys.argv[1]
system_name  = sys.argv[2]

def get_our_gold(hter_file):
    hter = []
    for i,lines in enumerate(open(hter_file)):
        hter.append(float(lines.strip()))
    return hter

def get_index(i,y_hat):
    y_hat_ = sorted(enumerate(y_hat),key=lambda x:x[1])
    for key,value in enumerate(y_hat_):
        if i == value[0]:
            return key+1

#获得预测值
our = get_our_gold(hter_file)
#生成文件
with open(hter_file+".csv", 'w') as _fout:
            for i, _y in enumerate(our):
                print >> _fout,  "%s\t%d\t%f\t%d" % (system_name,i+1,_y,get_index(i,our))

if __name__ == '__main__':
    pass
