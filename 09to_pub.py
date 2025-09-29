import os
import glob
import shutil

#data05/xco/shared_graph.tsv
target="data05/**/shared_graph.tsv"
os.makedirs("data06/",exist_ok=True)
for filename in glob.glob(target,recursive=True):
    dname=os.path.dirname(filename)
    name=os.path.basename(dname)
    out="data06/"+name+".graph.tsv"
    print(out)
    shutil.copy(filename,out)
