import os
import glob

def save_degree(filename, out_filename, out_dist_filename):
    degree_dist={}
    prev_node=None
    deg_count=0
    with open(out_filename,"w") as fp:
        for line in open(filename):
            arr=line.strip().split("\t")
            key=int(arr[0])
            if key != prev_node:
                deg_count+=1
                fp.write(str(key))
                fp.write("\t")
                fp.write(str(deg_count))
                fp.write("\n")
                if deg_count not in degree_dist:
                    degree_dist[deg_count]=0
                degree_dist[deg_count]+=1
                ###
                deg_count=0
                prev_node=key
            else:
                deg_count+=1
    
    with open(out_dist_filename,"w") as fp:
        for k,v in sorted(degree_dist.items()):
            fp.write(str(k))
            fp.write("\t")
            fp.write(str(v))
            fp.write("\n")
    
def run_out(name):
    os.makedirs("stat06_sort",exist_ok=True)
    filename="data06_sort/{}.graph.tsv".format(name)
    deg_filename="stat06_sort/{}.odegree.tsv".format(name)
    degdist_filename="stat06_sort/{}.odegree_dist.tsv".format(name)
    save_degree(filename, deg_filename, degdist_filename)

target="data06_sort/*.graph.tsv"
for filename in glob.glob(target,recursive=True):
    bname=os.path.basename(filename)
    name1,_ = os.path.splitext(bname)
    name,_ = os.path.splitext(name1)
    print("... running:",name)
    run_out(name)

