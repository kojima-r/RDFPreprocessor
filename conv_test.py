from rdflib import Graph
import glob
import os
from multiprocessing import Pool


#filename="./data01/uniprot/latest/uniprotkb_unreviewed_bacteria_proteobacteria_alphaproteobacteria_28211_13000000.rdf"

#filename="data01/clinvar/latest/ClinVarVariationRelease_00-latest_81.ttl"

filename="data01/biosampleplus/latest/biosample.671920305.ttl"
#for line in open(filename,newline="\n"):
#    a=line.strip().count("\r")
#    if a>0:
#        print(line)
#        #print(a)
#quit()
out_filename="test.nt"
g = Graph()
g.parse(filename, format="turtle")
print(len(g))
g.serialize(destination=out_filename,format="ntriples")




