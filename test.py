from  rdflib import Graph
import gzip
filename="data01/pubtator/latest/gene2pubtatorcentral-aa.ttl"
g = Graph()
with open(filename, 'rb') as f_in:
    g.parse(f_in, format="turtle")
print(len(g))

out_filename="test.ttl"

#g.serialize(destination=out_filename,format="turtle")
g.serialize(destination=out_filename,format="ntriples")
