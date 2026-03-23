import os
import glob
   
def run_out(name):
    filename="data06_uniq/{}.init_node.tsv".format(name)
    deg_filename="stat06_uniq/{}.odegree.tsv".format(name)
    with open(filename, "w") as ofp:
        for line in open(deg_filename):
            arr=line.strip().split("\t")
            if int(arr[1])>1:
                ofp.write("\t".join(arr))
                ofp.write("\n")
    #save_degree(filename, deg_filename, degdist_filename)

target="stat06_uniq/*.odegree.tsv"
for filename in glob.glob(target,recursive=True):
    bname=os.path.basename(filename)
    name1,_ = os.path.splitext(bname)
    name,_ = os.path.splitext(name1)
    print("... running:",name)
    run_out(name)

