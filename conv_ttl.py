from rdflib import Graph
import glob
import os
from multiprocessing import Pool,TimeoutError

def get_timeout(p,args):
    try:
        print(p,args)
        return p.get(timeout=60*60)
    except TimeoutError:
        print("timeout:",args)
        return None
    except:
        print("error:",args)
        return None


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
    try:
        if mode=="xml" or mode=="rdf":
            conv_xml(filename, out_filename)
        else:
            conv(filename, out_filename)
    except:
        print("error:",argv)
        return None

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
    pool = Pool(n_jobs) # プロセス数を4に設定
    pids=[(pool.apply_async(run, (args,)),args) for args in data]
    results=[get_timeout(p,args) for p,args in pids]

    #p.map(run, data)


