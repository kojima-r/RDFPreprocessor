import glob
import os
import joblib

def get_node_dict(filename):
    fp=open(filename,buffering=16*1024*1024)
    out=dict()
    for line in fp:
        line=line.strip()
        if len(line)>0:
            arr=line.split("\t")
            nid, nt, node =arr
            key=node
            out[key]=0
    return out

target="data05/**/node.tsv"
#src_filename="data05/bgee/node.tsv"
#out_filename="data05/bgee/shared_node.tsv"
skip_exist=False
for src_filename in glob.glob(target,recursive=True):
    #print(src_filename)
    dname=os.path.dirname(src_filename)
    out_filename=dname+"/shared_node.tsv"
    print(out_filename)

    if skip_exist and os.path.isfile(out_filename):
        print("[EXIST]", out_filename)
    else:
        node_dict = get_node_dict(src_filename)
        for filename in glob.glob(target,recursive=True):
            print(">>",filename)
            if filename != src_filename:
                fp=open(filename)
                for line in fp:
                    line=line.strip()
                    if len(line)>0:
                        arr=line.split("\t")
                        nid, nt, node =arr
                        key=node
                        if key in node_dict:
                            node_dict[key]+=1

        ofp=open(out_filename,"w")

        for key,val in node_dict.items():
            if val>0:
                ofp.write(key+"\t"+str(val))
                ofp.write("\n")

