import glob
import os
import joblib

def get_shared_edge_mapping(filename):
    global_mapping={}
    with open(filename,buffering=16*1024*1024) as fp:
        for line in fp:
            line=line.strip()
            arr=line.split("\t")
            #geid, eid,et,e =arr
            global_mapping[int(arr[1])]=int(arr[0])
    return global_mapping

target="data05/**/shared_edge.tsv"
out="data05/all_edge.tsv"
data={}
for filename in glob.glob(target,recursive=True):
    dname=os.path.dirname(filename)
    # shared_edge.tsv
    print(">>",filename)
    #edge_mapping = get_shared_edge_mapping(filename)
    global_mapping={}
    with open(filename,buffering=16*1024*1024) as fp:
        for line in fp:
            line=line.strip()
            arr=line.split("\t")
            geid, eid,et,e =arr
            if not geid in data:
                data[int(geid)]=[et,e]
with open(out,"w") as ofp:
    for k,v in sorted(data.items()):
        ofp.write("\t".join([str(k)]+v))
        ofp.write("\n")

