import glob
import os
import joblib

def get_shared_node_dict(target):
    out={}
    count_snode={}
    for filename in glob.glob(target,recursive=True):
        fp=open(filename,buffering=16*1024*1024)
        count_snode[filename]=0
        for line in fp:
            line=line.strip()
            if len(line)>0:
                arr=line.split("\t")
                node =arr[0]
                out[node]=len(out)
                count_snode[filename]+=1
    return out, count_snode

import mmap
def count_lines(filename):
    cnt = 0
    with open(filename, "r+") as f:
        mm = mmap.mmap(f.fileno(), 0)
        for line in iter(mm.readline, b""):
            if line.strip():
                cnt += 1
        mm.close()
    return cnt

def conv(filename,out_filename, shared_nodes, global_index):
    global_cnt=0
    local_cnt=0
    print(">>",out_filename)
    ofp = open(out_filename, 'w')
    for line in open(filename):
        line=line.strip()
        if len(line)>0:
            arr=line.split("\t")
            nid, nt, node =arr
            key=node
            if key in shared_nodes:
                ofp.write(str(nid)+"\t"+str(shared_nodes[key]))
                ofp.write("\n")
                global_cnt+=1
            else:
                ofp.write(str(nid)+"\t"+str(global_index))
                ofp.write("\n")
                global_index+=1
                local_cnt+=1

    return global_cnt, local_cnt, global_index 
target="data05/**/shared_node.tsv"
shared_nodes, count_snode=get_shared_node_dict(target)
global_index=len(shared_nodes)

print("#shared_nodes:", global_index)
print("#shared_nodes:", count_snode)

target="data05/**/node.tsv"
results={}
for filename in glob.glob(target,recursive=True):
    cnt=0
    dname=os.path.dirname(filename)
    k=dname+"/shared_node.tsv"
    out_filename=dname+"/node.global.tsv"
    #cnt=count_lines(filename)
    
    #cnt = sum(1 for line in open(filename) if line not in ("\n", ""))
    
    #for line in open(filename):
    #    line=line.strip()
    #    if len(line)>0:
    #        cnt+=1
    temp=global_index
    global_cnt, local_cnt, global_index = conv(filename,out_filename, shared_nodes, global_index)
    print(filename, "=>",local_cnt,"/",global_cnt)
    results[dname]={
            "shared_count":count_snode[k],
            "global_count":global_cnt,
            "local_count":local_cnt,
            "start_global_index":temp,
            "end_global_index":global_index,
            }

import json
new_json = open('data05/info.json', 'w')
json.dump(results, new_json)

