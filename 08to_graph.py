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

def get_shared_node_mapping(filename):
    global_mapping={}
    with open(filename,buffering=16*1024*1024) as fp:
        for line in fp:
            line=line.strip()
            arr=line.split("\t")
            #local=>global
            global_mapping[int(arr[0])]=int(arr[1])
    return global_mapping

target="data05/**/shared_edge.tsv"

for filename in glob.glob(target,recursive=True):
    dname=os.path.dirname(filename)

    print(">>",filename)
    edge_mapping = get_shared_edge_mapping(filename)
    node_filename=dname+"/node.global.tsv"
    print(">>",node_filename)
    node_mapping = get_shared_node_mapping(node_filename)
    g_filename=dname+"/graph.tsv"
    out_filename=dname+"/shared_graph.tsv"
    print(">>>>",out_filename)
    with open(out_filename,"w") as ofp:
        for line in open(g_filename):
            arr=line.split("\t")
            n1, e, n2 =arr
            n1i=node_mapping[int(n1)]
            n2i=node_mapping[int(n2)]
            ei=edge_mapping[int(e)]
            ofp.write("\t".join(map(str,[n1i,ei,n2i])))
            ofp.write("\n")
    


#for k,v in all_edge.items():
#    if len(v)>1:
#        print(k,len(v))

