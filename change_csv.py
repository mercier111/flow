import os 
import pandas as pd 


inpath = 'dataset/{}_S{}.csv'
outpath = 'dataset/_{}_S{}.csv'

def change_csv(inpath, outpath):
    with open(inpath) as r:
        with open(outpath, 'w+') as w:
            for line in r:
                w.write(line[1:-2] + '\n')

for i in ['805', '809', '814']:
    for j in ['train', 'test']:
        change_csv(inpath.format(j, i), outpath.format(j, i))