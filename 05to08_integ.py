"""
05to08_integ.py
パイプライン 05-08 を一本化したストリーミング実装。

入力:
    <input_dir>/<dataset>/**/*.tsv
    各 TSV は 6 列: nt1\tnt2\tnt3\tn1\tn2\tn3

出力 (大規模ファイル対応のためすべてストリーム書き出し):
    <output_dir>/node.tsv               全データセット横断のノード対応表
                                        (global_node_id, node_type, node_string)
    <output_dir>/edge.tsv               全データセット横断のエッジ対応表
                                        (global_edge_id, edge_type, edge_string)
    <output_dir>/<dataset>/graph.tsv    (n1_gid, e_gid, n2_gid)
    <output_dir>/<dataset>/literal.tsv  (n1_gid, nt1, nt2, nt3, n1, n2, n3)
    <output_dir>/info.json              データセットごとの統計情報

vocab バックエンド:
    --vocab-backend memory   Python dict (default, 高速だが TB 級では破綻)
    --vocab-backend lmdb     LMDB を使った disk-backed KV (TB 級対応, 速度低下)

実行例:
    python 05to08_integ.py
    python 05to08_integ.py --input-dir data02 --output-dir data05_integ
    python 05to08_integ.py --vocab-backend lmdb --vocab-dir data05_integ/.vocab
    python 05to08_integ.py --vocab-backend lmdb --map-size $((2 * 1024**4))
"""
import argparse
import glob
import json
import os
import shutil
import sys
import time


class MemVocab:
    """Python dict ベースのインメモリ vocab。"""

    def __init__(self):
        self._d = {}

    def get(self, key):
        return self._d.get(key)

    def add(self, key):
        i = len(self._d)
        self._d[key] = i
        return i

    def __len__(self):
        return len(self._d)

    def close(self):
        pass


class LmdbVocab:
    """
    LMDB ベースの disk-backed vocab。
    - key: utf-8 エンコードしたノード/エッジ文字列
    - value: 8byte little-endian 符号なし整数 (global ID)
    - 書き込みは 1 本の write txn で行い、batch_size 件ごとに commit。
    - ID は連番 (0 から) で `self._count` が次に割り当てる ID を保持。
    """

    def __init__(self, path, map_size, batch_size=200_000, sync=False):
        import lmdb
        self._lmdb = lmdb
        os.makedirs(path, exist_ok=True)
        self.path = path
        self.env = lmdb.open(
            path,
            map_size=map_size,
            subdir=True,
            max_dbs=1,
            sync=sync,
            metasync=sync,
            writemap=True,
            map_async=True,
            lock=True,
            readahead=False,
        )
        with self.env.begin() as txn:
            self._count = txn.stat()["entries"]
        self.batch_size = batch_size
        self._txn = self.env.begin(write=True, buffers=True)
        self._pending = 0

    def get(self, key):
        v = self._txn.get(key.encode("utf-8"))
        if v is None:
            return None
        return int.from_bytes(bytes(v), "little", signed=False)

    def add(self, key):
        i = self._count
        self._txn.put(key.encode("utf-8"), i.to_bytes(8, "little", signed=False))
        self._count += 1
        self._pending += 1
        if self._pending >= self.batch_size:
            self._flush()
        return i

    def _flush(self):
        self._txn.commit()
        self._txn = self.env.begin(write=True, buffers=True)
        self._pending = 0

    def __len__(self):
        return self._count

    def close(self):
        if self._pending > 0:
            self._txn.commit()
        else:
            self._txn.abort()
        self.env.sync(True)
        self.env.close()


def make_vocab(backend, name, vocab_dir, map_size, fresh):
    if backend == "memory":
        return MemVocab()
    if backend == "lmdb":
        path = os.path.join(vocab_dir, name)
        if fresh and os.path.isdir(path):
            shutil.rmtree(path)
        return LmdbVocab(path, map_size=map_size)
    raise ValueError(f"unknown backend: {backend}")


def conv_bnode(node_name, filename):
    return os.path.basename(filename) + "_" + node_name


def process_dataset(
    ds_path,
    ds_out_dir,
    node_vocab,
    edge_vocab,
    node_out,
    edge_out,
    buf_size,
):
    os.makedirs(ds_out_dir, exist_ok=True)
    graph_path = os.path.join(ds_out_dir, "graph.tsv")
    literal_path = os.path.join(ds_out_dir, "literal.tsv")

    n_triples = 0
    n_literals = 0
    n_skipped = 0
    nodes_before = len(node_vocab)
    edges_before = len(edge_vocab)

    with open(graph_path, "w", buffering=buf_size) as graph_out, \
         open(literal_path, "w", buffering=buf_size) as literal_out:

        gw = graph_out.write
        lw = literal_out.write
        nw = node_out.write
        ew = edge_out.write
        n_get = node_vocab.get
        n_add = node_vocab.add
        e_get = edge_vocab.get
        e_add = edge_vocab.add

        tsv_files = sorted(
            glob.glob(os.path.join(ds_path, "**", "*.tsv"), recursive=True)
        )
        for filename in tsv_files:
            print(">>", filename, flush=True)
            with open(filename, "r", encoding="utf-8",
                      errors="replace", buffering=buf_size) as fp:
                for line in fp:
                    arr = line.rstrip("\n").split("\t")
                    if len(arr) != 6:
                        n_skipped += 1
                        continue
                    nt1, nt2, nt3, n1, n2, n3 = arr

                    if nt1 == "BNode":
                        n1 = conv_bnode(n1, filename)

                    n1_i = n_get(n1)
                    if n1_i is None:
                        n1_i = n_add(n1)
                        nw(f"{n1_i}\t{nt1}\t{n1}\n")

                    if nt3 == "Literal":
                        lw(f"{n1_i}\t{nt1}\t{nt2}\t{nt3}\t{n1}\t{n2}\t{n3}\n")
                        n_literals += 1
                    else:
                        if nt3 == "BNode":
                            n3 = conv_bnode(n3, filename)

                        n3_i = n_get(n3)
                        if n3_i is None:
                            n3_i = n_add(n3)
                            nw(f"{n3_i}\t{nt3}\t{n3}\n")

                        n2_i = e_get(n2)
                        if n2_i is None:
                            n2_i = e_add(n2)
                            ew(f"{n2_i}\t{nt2}\t{n2}\n")

                        gw(f"{n1_i}\t{n2_i}\t{n3_i}\n")
                        n_triples += 1

    return {
        "n_triples": n_triples,
        "n_literals": n_literals,
        "n_skipped_lines": n_skipped,
        "n_new_nodes": len(node_vocab) - nodes_before,
        "n_new_edges": len(edge_vocab) - edges_before,
        "global_node_count_after": len(node_vocab),
        "global_edge_count_after": len(edge_vocab),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Integrated 05-08 streaming pipeline."
    )
    parser.add_argument("--input-dir", default="data02",
                        help="入力ディレクトリ (default: data02)")
    parser.add_argument("--output-dir", default="data05_integ",
                        help="出力ディレクトリ (default: data05_integ)")
    parser.add_argument("--buf-size", type=int, default=16 * 1024 * 1024,
                        help="I/O buffer size in bytes (default: 16MiB)")
    parser.add_argument("--dataset", action="append", default=None,
                        help="特定のデータセット名のみ処理 (複数指定可)")
    parser.add_argument("--vocab-backend", choices=["memory", "lmdb"],
                        default="memory",
                        help="vocab バックエンド (default: memory)")
    parser.add_argument("--vocab-dir", default=None,
                        help="LMDB の格納先 (default: <output_dir>/.vocab)")
    parser.add_argument("--map-size", type=int, default=1 << 40,
                        help="LMDB の map_size (bytes, default: 1TiB)")
    parser.add_argument("--keep-vocab", action="store_true",
                        help="LMDB ディレクトリを実行前に削除しない (resume 用)")
    args = parser.parse_args()

    if not os.path.isdir(args.input_dir):
        print(f"[ERROR] input dir not found: {args.input_dir}", file=sys.stderr)
        sys.exit(1)

    os.makedirs(args.output_dir, exist_ok=True)

    if args.dataset:
        ds_paths = [os.path.join(args.input_dir, d) for d in args.dataset]
    else:
        ds_paths = sorted(
            p for p in glob.glob(os.path.join(args.input_dir, "*"))
            if os.path.isdir(p)
        )

    if not ds_paths:
        print("[ERROR] no datasets found.", file=sys.stderr)
        sys.exit(1)

    vocab_dir = args.vocab_dir or os.path.join(args.output_dir, ".vocab")
    fresh = (args.vocab_backend == "lmdb") and (not args.keep_vocab)
    print(f"[vocab] backend={args.vocab_backend} dir={vocab_dir} "
          f"map_size={args.map_size} fresh={fresh}", flush=True)
    node_vocab = make_vocab(args.vocab_backend, "node", vocab_dir,
                            args.map_size, fresh)
    edge_vocab = make_vocab(args.vocab_backend, "edge", vocab_dir,
                            args.map_size, fresh)

    info = {}

    node_out_path = os.path.join(args.output_dir, "node.tsv")
    edge_out_path = os.path.join(args.output_dir, "edge.tsv")

    node_open_mode = "a" if args.keep_vocab else "w"
    edge_open_mode = node_open_mode

    t0 = time.time()
    try:
        with open(node_out_path, node_open_mode, buffering=args.buf_size) as node_out, \
             open(edge_out_path, edge_open_mode, buffering=args.buf_size) as edge_out:

            for i, ds_path in enumerate(ds_paths):
                ds_name = os.path.basename(ds_path.rstrip("/"))
                ds_out_dir = os.path.join(args.output_dir, ds_name)
                print(f"=== [{i + 1}/{len(ds_paths)}] {ds_name} ===", flush=True)
                t_ds = time.time()
                stats = process_dataset(
                    ds_path, ds_out_dir,
                    node_vocab, edge_vocab,
                    node_out, edge_out,
                    args.buf_size,
                )
                stats["elapsed_sec"] = round(time.time() - t_ds, 2)
                info[ds_name] = stats
                print(
                    f"... {ds_name}: triples={stats['n_triples']} "
                    f"literals={stats['n_literals']} "
                    f"new_nodes={stats['n_new_nodes']} "
                    f"new_edges={stats['n_new_edges']} "
                    f"skipped={stats['n_skipped_lines']} "
                    f"({stats['elapsed_sec']}s)",
                    flush=True,
                )
    finally:
        node_vocab.close()
        edge_vocab.close()

    summary = {
        "total_nodes": len(node_vocab) if hasattr(node_vocab, "__len__") else None,
        "total_edges": len(edge_vocab) if hasattr(edge_vocab, "__len__") else None,
        "elapsed_sec": round(time.time() - t0, 2),
        "vocab_backend": args.vocab_backend,
        "datasets": info,
    }
    with open(os.path.join(args.output_dir, "info.json"), "w") as fp:
        json.dump(summary, fp, indent=2, ensure_ascii=False)

    print("=== summary ===")
    print(f"total nodes: {summary['total_nodes']}")
    print(f"total edges: {summary['total_edges']}")
    print(f"elapsed   : {summary['elapsed_sec']}s")


if __name__ == "__main__":
    main()
