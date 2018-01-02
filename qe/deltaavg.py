#coding:utf8

import sys
import numpy
import math

ref = sys.argv[1]
pred = sys.argv[2]


def read_file(file_name):
    score = [0]
    rank = [0]
    
    with open(file_name) as fp:
        for lines in fp:
            lines = lines.strip().split('\t')
            score.append(float(lines[2]))
            rank.append(int(lines[3]))
        
    return score,rank
    
ref_data = read_file(ref)
pred_data = read_file(pred)

#print ref_data
#print pred_data

V = [0]
V.extend(sorted(pred_data[0]))

N = len(ref_data[0])/2

#平均值
V_ = numpy.mean(pred_data[0])


deltaavg = [0,0]
length = len(ref_data[0])

for i in range(2,N+1):
    v_ = 0
    size = length/i
    
    for k in range(1,i):
        v_ += numpy.mean(V[1:k*size+1])

    result = v_/(i-1) -V_
    deltaavg.append(result)
       
deltaavg = sum(deltaavg)/(N-1)
print '%f' % abs(deltaavg)

















