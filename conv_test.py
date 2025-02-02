from rdflib import Graph
import glob
import os
from multiprocessing import Pool

mode="xml"

filename="./data01/uniprot/latest/uniprotkb_unreviewed_bacteria_proteobacteria_alphaproteobacteria_28211_13000000.rdf"
out_filename="test.ttl"
g = Graph()
g.parse(filename, format="xml")
print(len(g))
g.serialize(destination=out_filename,format="turtle")


