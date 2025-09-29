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
    print("[save]",out_filename)

conv("data03/owl/latest/owl.nt", "test.tsv")


