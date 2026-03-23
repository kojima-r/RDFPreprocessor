   
import duckdb
import glob
import os

#current_id=99999998
#for in open(target_filename):
def run(target_filename,outfilename): 
    con = duckdb.connect()
    con.execute("PRAGMA threads=8")
    con.execute("PRAGMA memory_limit='32GB'")
    con.execute("PRAGMA temp_directory='/tmp'")

    con.execute(f"""
COPY (
  SELECT *
  FROM read_csv(
    '{target_filename}',
    delim='\t',
    header=true,
    columns={{'c1':'BIGINT', 'c2':'STRING', 'c3':'STRING'}},
    sample_size=-1
  )
)
TO '{outfilename}'
(FORMAT PARQUET, COMPRESSION ZSTD);
""")

os.makedirs("data08_node",exist_ok=True)

for filename in glob.glob("data06/*.node.tsv"):
    bname = os.path.basename(filename)
    name,_ = os.path.splitext(bname)
    
    print(name)
    #target_filename="data06_uniq/bgee.graph.tsv"
    #outfilename="data08/bgee.graph.parquet"
    if os.path.getsize(filename)>10:
        run(filename, "data08_node/"+name+".parquet")

