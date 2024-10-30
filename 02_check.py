import glob
import os
from multiprocessing import Pool
from rdflib import Graph
mode="ttl"
def run(argv):
    filename = argv
    print(filename)
    #path=os.path.dirname(filename)
    g = Graph()
    g.parse(filename, format="turtle")
    print(filename, len(g))
 

if __name__ == "__main__":
    data=[]
    if mode=="xml":
        target="data01/**/*.xml"
    else:
        target="data01/**/*.ttl"
    for filename in glob.glob(target,recursive=True):
        data.append(filename)
    p = Pool(8) # プロセス数を4に設定
    p.map(run, data)


