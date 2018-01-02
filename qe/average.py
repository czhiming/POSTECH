#coding:utf8

import sys
import numpy

files = sys.argv[1:]

fps = [open(file_name) for file_name in files]

def get_hter(fp):
	hter = []
	for lines in fp:
		lines = float(lines.strip())
		hter.append(lines)

	return hter
    
all = []    
for fp in fps:
	all.append(get_hter(fp))

all = numpy.array(all)
result = all.mean(axis=0)

with open('average.hter.pred','w') as fp:
	for hter in result:
		fp.writelines(str(hter)+'\n')

[fp.close() for fp in fps]










