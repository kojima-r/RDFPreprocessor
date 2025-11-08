import glob
import os
import numpy as np
import json
def get_node_count():
    obj=json.load(open("data05/info.json"))
    out={}
    for k,v in obj.items():
        out[k]=v["global_count"]+v["local_count"]
    return out
def get_shared_node_mapping(filename,n_count):
    global_mapping=np.zeros(n_count+1,dtype=np.int64)
    with open(filename,buffering=16*1024*1024) as fp:
        for line in fp:
            line=line.strip()
            arr=line.split("\t")
            #local=>global
            k=int(arr[0])
            if k<=n_count:
                global_mapping[k]=int(arr[1])
            else:
                print("skip...:",k)
    return global_mapping

target="data05/**/node.tsv"
results={}
cnt_list=get_node_count()
for filename in glob.glob(target,recursive=True):
    cnt=0
    dname=os.path.dirname(filename)
    node_filename=dname+"/node.global.tsv"
    n_count=cnt_list[dname]
    print(filename, "=>",n_count)
    node_mapping = get_shared_node_mapping(node_filename, n_count)
    out_filename=dname+"/node_list.global.tsv"
    with open(out_filename,"w") as ofp:
        for line in open(filename):
            line=line.strip()
            if len(line)>0:
                arr=line.split("\t")
                n1=arr[0]
                n1i=node_mapping[int(n1)]
                ofp.write("\t".join([str(n1i)]+arr[1:]))
                ofp.write("\n")


