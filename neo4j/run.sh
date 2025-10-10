neo4j-admin database import full --overwrite-destination neo4j \
  --nodes=Entity=/data4/kojima/RDFPreprocessor/neo4j/node_id.tsv \
  --relationships=/data4/kojima/RDFPreprocessor/neo4j/dummy_head.tsv,"/data4/kojima/RDFPreprocessor/data06/.*\.graph\.tsv" \
  --delimiter="\t"
#neo4j-admin database import full --overwrite-destination graph.db \
#  --nodes=Entity=/data4/kojima/RDFPreprocessor/neo4j/node_id.tsv \
#  --relationships=/data4/kojima/RDFPreprocessor/neo4j/dummy_head.tsv /data4/kojima/RDFPreprocessor/data06/*.tsv \
#  --delimiter="\t"

#--id-type=STRING \
