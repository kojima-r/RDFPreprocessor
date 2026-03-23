import duckdb

con_n = duckdb.connect()
con_n.execute("PRAGMA threads=8")
con_n.execute("PRAGMA memory_limit='16GB'")
con_n.execute("PRAGMA temp_directory='/tmp'")
#s="506596419"
#s=99999998
#s=506596419
s=2914086210
#s=2969769
# Parquetを直接テーブル扱いでクエリできる
df = con_n.execute("""
SELECT *
FROM read_parquet('data08_node/*.node.parquet')
WHERE c1 = {}
LIMIT 20
""".format(s)).df()
print(df)
#FROM 'data08/bgee.graph.parquet'
