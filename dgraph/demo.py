# Create example split TSV files (UTF-8) for the earlier schema and flow.
from pathlib import Path
import gzip

base = Path("./data/")
base.mkdir(parents=True, exist_ok=True)

people1 = """id\tname\tage
P0001\t1\t34
P0002\t2\t28
P0003\t3\t41
P0004\t4\t35
P0005\t5\t52
""".strip()+"\n"

people2 = """id\tname\tage
P0006\t1\t29
P0007\t2\t33
P0008\t3\t47
P0009\t4\t26
P0010\t5\t45
""".strip()+"\n"

rels1 = """src_id\tfriend_id
P0001\tP0002
P0001\tP0003
P0002\tP0004
P0003\tP0005
P0004\tP0006
P0006\tP0001
""".strip()+"\n"

files = {
    base/"people_part-0001.tsv": people1,
    base/"people_part-0002.tsv": people2,
    base/"relationships_part-0001.tsv": rels1,
}

for p, content in files.items():
    p.write_text(content, encoding="utf-8")
    with gzip.open(str(p)+".gz", "wt", encoding="utf-8") as gz:
        gz.write(content)

sorted([str(p) for p in base.iterdir()])

