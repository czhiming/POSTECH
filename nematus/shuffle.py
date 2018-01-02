#-*- coding:utf8 -*-
'''
Created on Apr 27, 2017

@author: czm
'''
#随机打乱文件
import os
import sys
import random

from tempfile import mkstemp

def main(files):
    #创建临时文件，返回安全级别和临时文件的路径
    tf_os,tpath = mkstemp()
    tf = open(tpath,'w')
    
    fds = [open(ff) for ff in files]
    
    for l in fds[0]:
        lines = [l.strip()] + [ff.readline().strip() for ff in fds[1:]]
        print >> tf,"|||".join(lines)
    
    [ff.close() for ff in fds]
    tf.close()
    
    tf = open(tpath,'r')
    lines = tf.readlines()
    random.shuffle(lines)
    
    fds = [open(ff+'.shuf','w') for ff in files]
    
    for l in lines:
        s = l.strip().split('|||')
        for ii,fd in enumerate(fds):
            print >>fd,s[ii]
    
    [ff.close() for ff in fds]
    
    os.remove(tpath) #删临时文件



if __name__ == '__main__':
    pass