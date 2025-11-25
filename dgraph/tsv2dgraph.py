from pathlib import Path
import sys, csv, json, argparse, re

def sanitize_pred(name: str) -> str:
    # Dgraph predicate must not contain spaces etc. Replace non-word chars with underscore.
    return re.sub(r"[^\w]", "_", name)

def parse_cols_list(s: str):
    if not s:
        return set()
    return set([sanitize_pred(x.strip()) for x in s.split(",") if x.strip()])

def main():
    ap = argparse.ArgumentParser(description="Convert TSV to Dgraph JSON (for /mutate).")
    ap.add_argument("tsv", help="Input TSV file (first line headers). Use '-' for stdin.")
    ap.add_argument("--type-name", default="Row", help="dgraph.type to set on each node (default: Row)")
    ap.add_argument("--id-col", default=None, help="Name of the TSV column to map as 'external_id'")
    ap.add_argument("--int-cols", default="", help="Comma-separated column names to cast to int")
    ap.add_argument("--float-cols", default="", help="Comma-separated column names to cast to float")
    ap.add_argument("--limit", type=int, default=None, help="Stop after N rows (debugging)")
    args = ap.parse_args()

    int_cols = parse_cols_list(args.int_cols)
    float_cols = parse_cols_list(args.float_cols)
    id_col = sanitize_pred(args.id_col) if args.id_col else None

    # Open input
    fin = sys.stdin if args.tsv == "-" else open(args.tsv, "r", newline="", encoding="utf-8")
    with fin:
        rdr = csv.DictReader(fin, delimiter="\t")
        # Sanitize header names
        headers = [sanitize_pred(h) for h in rdr.fieldnames]
        # Rebuild reader with sanitized fieldnames
        fin.seek(0)
        raw = csv.reader(fin, delimiter="\t")
        first = next(raw)
        # Create mapping from raw headers to sanitized
        mapping = {raw_h: sanitize_pred(raw_h) for raw_h in first}
        # Build an iterator that yields dicts with sanitized keys
        def rows():
            for r in raw:
                d = {}
                for raw_h, v in zip(first, r):
                    key = mapping[raw_h]
                    d[key] = v
                yield d

        nodes = []
        for i, row in enumerate(rows(), 1):
            node = {"uid": f"_:r{i}", "dgraph.type": [args.type_name]}
            for k, v in row.items():
                if v == "":
                    continue
                key = k
                # Special-case id_col -> external_id
                if id_col and key == id_col:
                    key = "external_id"
                # Cast numeric if requested
                if key in int_cols:
                    try:
                        node[key] = int(v)
                    except ValueError:
                        node[key] = v  # fallback to string
                elif key in float_cols:
                    try:
                        node[key] = float(v)
                    except ValueError:
                        node[key] = v
                else:
                    node[key] = v
            nodes.append(node)
            if args.limit and i >= args.limit:
                break

        out = {"set": nodes}
        json.dump(out, sys.stdout, ensure_ascii=False)
        sys.stdout.write("\n")
    
if __name__ == "__main__":
    main()
