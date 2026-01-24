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
    ##
    filename=dname+"/node_list.global.tsv"
    out="data06/"+name+".node.tsv"
    shutil.copy(filename,out)
    ##
    filename=dname+"/literal.global.tsv"
    out="data06/"+name+".literal.tsv"
    shutil.copy(filename,out)
