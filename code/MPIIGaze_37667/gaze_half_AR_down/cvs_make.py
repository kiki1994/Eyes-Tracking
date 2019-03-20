# -*- coding: utf-8 -*-

import os
import csv


with open('record.txt','r') as f:
    lines = f.read().splitlines()
    per = len(lines)
    print(per)
    for line in  [i for i in lines if i !="\n" ]:  
        li = []      
        strs = line.split(' ')[0]      
        acc = line.split(' ')[1]
        #iteration = line.split(' ')[2]
        #epoch = line.split(' ')[3]

        li.extend([strs,acc])
        print(li)              
        out = open('data.csv','a',newline='')
        csv_write = csv.writer(out,dialect = 'excel')
        csv_write.writerow(li)