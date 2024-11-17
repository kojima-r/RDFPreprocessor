from rdflib import Graph
import glob
import os
from multiprocessing import Pool

def conv(filename, out_filename):
    g = Graph()
    g.parse(filename, format="ntriples")
    with open(out_filename,"w") as ofp:
        for s, p, o in g:
            ss=str(s).replace("\r\n","  ").replace("\n","  ").replace("\t","  ")
            pp=str(p).replace("\r\n","  ").replace("\n","  ").replace("\t","  ")
            oo=str(o).replace("\r\n","  ").replace("\n","  ").replace("\t","  ")
            line="\t".join([str(type(s).__name__),str(type(p).__name__),  str(type(o).__name__), ss,pp,oo])
            ofp.write(line)
            ofp.write("\n")

def run(argv):
    filename, out_filename = argv
    print(">>",filename)
    conv(filename, out_filename)


if __name__ == "__main__":
    data=[]
    target="data04/**/*.nt"
    for filename in glob.glob(target,recursive=True):
        path=os.path.dirname(filename)
        name=os.path.basename(filename)
        path1="data05"+path[6:]
        os.makedirs(path1,exist_ok=True)
        filename = path+"/"+name
        name_, _=os.path.splitext(name)
        out_filename = path1+"/"+name_+".tsv"
        print(filename, out_filename)
        data.append((filename, out_filename))
    p = Pool(8)
    p.map(run, data)
