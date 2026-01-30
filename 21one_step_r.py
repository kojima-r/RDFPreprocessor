import duckdb

con = duckdb.connect()
con.execute("PRAGMA threads=8")
con.execute("PRAGMA memory_limit='16GB'")
con.execute("PRAGMA temp_directory='/tmp'")
#s="506596419"
#s=99999998
#s=506596419
#s=3171575152
s=2
#s=2969769
# Parquetを直接テーブル扱いでクエリできる
df = con.execute("""
SELECT *
FROM read_parquet('data08/*.graph.parquet')
WHERE c3 = {}
LIMIT 20
""".format(s)).df()
print(df)
#FROM 'data08/bgee.graph.parquet'
