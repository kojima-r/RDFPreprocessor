from rdflib import Graph
import glob
import os
from multiprocessing import Pool

mode="ttl"#"xml"
def conv(filename, out_filename):
    g = Graph()
    g.parse(filename, format="turtle")
    print(len(g))
    g.serialize(destination=out_filename,format="ntriples")

def run(argv):
    filename, out_filename = argv
    print(">>",filename)
    conv(filename, out_filename)


if __name__ == "__main__":
    data=[]
    target="data03/**/*.ttl"
    for filename in glob.glob(target,recursive=True):
        path=os.path.dirname(filename)
        name=os.path.basename(filename)
        path1="data04"+path[6:]
        os.makedirs(path1,exist_ok=True)
        filename = path+"/"+name
        name_, _=os.path.splitext(name)
        out_filename = path1+"/"+name_+".nt"
        print(filename, out_filename)
        data.append((filename, out_filename))
        #if mode=="xml":
        #else:
        #    #name_, _=os.path.splitext(name_)
        #    out_filename = path1+"/"+name_+".ttl"
        #    print(filename, out_filename)
    p = Pool(8) # プロセス数を4に設定
    p.map(run, data)
