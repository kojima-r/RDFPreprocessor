import glob
import os
import joblib

def get_shared_edge_dict(target):
    out={}
    global_mapping={}
    for filename in glob.glob(target,recursive=True):
        fp=open(filename,buffering=16*1024*1024)
        for line in fp:
            line=line.strip()
            if len(line)>0:
                arr=line.split("\t")
                eid,et,e =arr
                if e not in out:
                    out[e]={}
                    global_mapping[e]=len(out)
                out[e][filename]=(eid, et, global_mapping[e])
    return out

def extract_edge_map(all_edge, filename):
    e_data=[]
    for k,v in all_edge.items():
        if filename in v:
            local_eid=v[filename][0]
            et=v[filename][1]
            e=k
            global_eid=v[filename][2]
            e_data.append([global_eid, local_eid, et, e])
    return e_data

target="data05/**/edge.tsv"
all_edge =  get_shared_edge_dict(target)

for filename in glob.glob(target,recursive=True):
    e_data=extract_edge_map(all_edge, filename)
    
    dname=os.path.dirname(filename)
    out_filename=dname+"/shared_edge.tsv"
    print(out_filename)
    with open(out_filename,"w") as ofp:
        for line in e_data:
            ofp.write("\t".join(map(str,line)))
            ofp.write("\n")
    


#for k,v in all_edge.items():
#    if len(v)>1:
#        print(k,len(v))

