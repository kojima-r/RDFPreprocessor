import glob
import os
from multiprocessing import Pool

def fix_ttl(filename, out_filename):
    shared_data=set()
    count=0
    line_cnt=0
    ofp=open(out_filename+".{:04d}.fix_ttl2".format(count),"w") 
    prev_line=None
    for line in open(filename):
        ### 1 byte x 30(10-60) characters  x  50M
        if line_cnt>50000000 and line[0]!=" " and prev_line.strip()=="":
            count+=1
            line_cnt=0
            o=out_filename+".{:04d}.fix_ttl2".format(count)
            print(">>",o)
            ofp=open(o,"w") 
            for e in shared_data:
                ofp.write(e)
                ofp.write("\n")
                line_cnt+=1

        el=line.strip()
        if line[:7]=="@prefix":
            if el not in shared_data:
                shared_data.add(el)
                ofp.write(line)
                line_cnt+=1
        else:
            if len(el)>0:
                ofp.write(line)
                line_cnt+=1
        prev_line=line

    print(len(shared_data))
    for el in shared_data:
        print(el)

def run(argv):
    filename, out_filename = argv
    print(">>",filename)
    fix_ttl(filename, out_filename)
    print(">>>",out_filename)


if __name__ == "__main__":
    filelist=[
            "data01/expressionatlas/latest/E-MTAB-2706.ttl",
            "data01/biosample/latest/bioschemas.ttl",
            "data01/expressionatlas/latest/E-MTAB-2770.ttl",
            "data01/expressionatlas/latest/E-MTAB-4748.ttl"]
    for filename in filelist:
        path=os.path.dirname(filename)
        name=os.path.basename(filename)
        name_, _ =os.path.splitext(name)
        out_filename = path+"/"+name_
        fix_ttl(filename, out_filename)

       
