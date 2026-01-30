import glob
import os

from multiprocessing import Pool

def run(filename1,filename2, ofp, name2):
    with open(filename1) as fp1:
        with open(filename2) as fp2:
            try:
                l1=next(fp1)
                l2=next(fp1)
                while fp1 or fp2:
                    arr1=l1.strip().split("\t")
                    arr2=l2.strip().split("\t")
                    key1=arr1[0]
                    key2=arr2[0]
                    if key1>key2:
                        l2=next(fp2)
                    elif key1<key2:
                        l1=next(fp1)
                    else:
                        #print("share:",arr1,arr2)
                        ofp.write("\t".join(arr2))
                        ofp.write("\t")
                        ofp.write(name2)
                        ofp.write("\n")
                        l2=next(fp2)
            except StopIteration: # EOF
                pass

target="data06_uniq/*.graph.tsv"
out_path="data07/"
os.makedirs(out_path,exist_ok=True)

def get_name(filename):
    bname=os.path.basename(filename)
    name1,_ = os.path.splitext(bname)
    name,_ = os.path.splitext(name1)
    return name

def run_job(src_filename):
    src_name=get_name(src_filename)
    out_filename=out_path+src_name+".share.tsv"
    with open(out_filename,"w") as ofp:
        for dest_filename in glob.glob(target,recursive=True):
            if src_filename==dest_filename:
                continue
            dest_name=get_name(dest_filename)
            print("... running:",src_name,"=>",dest_name)
            run(src_filename,dest_filename,ofp,dest_name)
  

if __name__ == "__main__":
    data=[]
    for src_filename in glob.glob(target,recursive=True):
        data.append(src_filename)
    p = Pool(8)
    p.map(run_job, data)
