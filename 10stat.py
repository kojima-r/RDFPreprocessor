import os
import glob

def get_out_degree(filename):
    out_degree={}
    for line in open(filename):
        arr=line.strip().split("\t")
        key=int(arr[0])
        if key not in out_degree:
            out_degree[key]=0
        out_degree[key]+=1
    return out_degree

def get_in_degree(filename):
    in_degree={}
    for line in open(filename):
        arr=line.strip().split("\t")
        key=int(arr[2])
        if key not in in_degree:
            in_degree[key]=0
        in_degree[key]+=1
    return in_degree

def get_degree_dist(deg_filename):
    out_degree={}
    for line in open(deg_filename):
        arr=line.strip().split("\t")
        k=int(arr[1])
        if k not in out_degree:
            out_degree[k]=0
        out_degree[k]+=1
    return out_degree

def save_degree(out_filename, degree_dict):
    with open(out_filename,"w") as fp:
        for k,v in degree_dict.items():
            fp.write(str(k))
            fp.write("\t")
            fp.write(str(v))
            fp.write("\n")

def save_degree_dist(out_filename, degree_dict):
    save_degree(out_filename, degree_dict)

def run_out(name):
    os.makedirs("stat06",exist_ok=True)
    filename="data06/{}.graph.tsv".format(name)
    deg_filename="stat06/{}.odegree.tsv".format(name)
    degdist_filename="stat06/{}.odegree_dist.tsv".format(name)
    odeg=get_out_degree(filename)
    save_degree(deg_filename, odeg)
    odeg_dist=get_degree_dist(deg_filename)
    save_degree(degdist_filename, odeg_dist)
 
def run_in(name):
    os.makedirs("stat06",exist_ok=True)
    filename="data06/{}.graph.tsv".format(name)
    deg_filename="stat06/{}.idegree.tsv".format(name)
    degdist_filename="stat06/{}.idegree_dist.tsv".format(name)
    odeg=get_in_degree(filename)
    save_degree(deg_filename, odeg)
    odeg_dist=get_degree_dist(deg_filename)
    save_degree(degdist_filename, odeg_dist)
       
target="data06/*.graph.tsv"
for filename in glob.glob(target,recursive=True):
    bname=os.path.basename(filename)
    name1,_ = os.path.splitext(bname)
    name,_ = os.path.splitext(name1)
    print("... running:",name)
    run_out(name)
    run_in(name)

