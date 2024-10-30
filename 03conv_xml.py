from rdflib import Graph
import glob
import os
from multiprocessing import Pool

mode="xml"
def conv(filename, out_filename):
    g = Graph()
    g.parse(filename, format="turtle")
    print(len(g))
    g.serialize(destination=out_filename,format="turtle")

def conv_xml(filename, out_filename):
    g = Graph()
    g.parse(filename, format="xml")
    print(len(g))
    g.serialize(destination=out_filename,format="turtle")

def run(argv):
    filename, out_filename = argv
    print(">>",filename)
    if mode=="xml":
        conv_xml(filename, out_filename)
    else:
        conv(filename, out_filename)


if __name__ == "__main__":
    data=[]
    if mode=="xml":
        target="data01/**/*.xml"
    else:
        target="data01/**/*.fix_ttl2"
    for filename in glob.glob(target,recursive=True):
        path=os.path.dirname(filename)
        name=os.path.basename(filename)
        path1="data03"+path[6:]
        os.makedirs(path1,exist_ok=True)
        filename = path+"/"+name
        name_, _=os.path.splitext(name)
        out_filename = path1+"/"+name_+".ttl"
        print(filename, out_filename)
        data.append((filename, out_filename))
        #if mode=="xml":
        #else:
        #    #name_, _=os.path.splitext(name_)
        #    out_filename = path1+"/"+name_+".ttl"
        #    print(filename, out_filename)
        #    data.append((filename, out_filename))
    p = Pool(8) # プロセス数を4に設定
    p.map(run, data)


