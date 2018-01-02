#coding:utf8


import sys
import re

if len(sys.argv) < 3:
    sys.exit(0)

file_name = sys.argv[1]
test = sys.argv[2]

pattern = re.compile(r'<seg id="\d+">(.*?)</seg>')

with open(test,'w') as fp:
    for lines in open(file_name):
        lines = lines.strip()
        content = pattern.findall(lines)
        if content != []:
            fp.writelines(content[0]+'\n')















