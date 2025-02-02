from rdflib import Graph
import glob
import os
from multiprocessing import Pool

def conv(filename, out_filename):
    g = Graph()
    g.parse(filename, format="turtle")
    print(len(g))
    g.serialize(destination=out_filename,format="ntriples")

def conv_xml(filename, out_filename):
    g = Graph()
    g.parse(filename, format="xml")
    print(len(g))
    g.serialize(destination=out_filename,format="ntriples")

def run(argv):
    filename, out_filename, mode = argv
    print(">>",filename)
    if mode=="xml" or mode=="rdf":
        conv_xml(filename, out_filename)
    else:
        conv(filename, out_filename)


def main_conv(mode, n_jobs):
    data=[]
    if mode=="xml":
        target="data01/**/*.xml"
    elif mode=="ttl2":
        target="data01/**/*.fix_ttl2"
    elif mode=="rdf":
        target="data01/**/*.rdf"
    else:
        target="data01/**/*.ttl"
    for filename in glob.glob(target,recursive=True):
        path=os.path.dirname(filename)
        name=os.path.basename(filename)
        path1="data03"+path[6:]
        os.makedirs(path1,exist_ok=True)
        filename = path+"/"+name
        name_, _=os.path.splitext(name)
        out_filename = path1+"/"+name_+".nt"
        if not os.path.isfile(out_filename):
            print(filename, out_filename)
            data.append((filename, out_filename, mode))
        else:
            print("[EXIST]", out_filename)
    p = Pool(n_jobs) # プロセス数を4に設定
    p.map(run, data)


