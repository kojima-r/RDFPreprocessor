import argparse
import json
import os
import random
import time
from typing import List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import IterableDataset, DataLoader


def str2bool(v: str) -> bool:
    if isinstance(v, bool):
        return v
    v = v.lower()
    if v in ("true", "1", "yes", "y"):
        return True
    if v in ("false", "0", "no", "n"):
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {v}")


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_mapping(path: str) -> dict:
    mapping = {}
    with open(path, "r", encoding="utf-8") as f:
        header = next(f, None)
        if header is None:
            return mapping
        for line in f:
            line = line.strip()
            if not line:
                continue
            raw_id, mapped_id = line.split("\t")
            mapping[raw_id] = int(mapped_id)
    return mapping


def count_lines(paths: List[str]) -> int:
    total = 0
    for path in paths:
        with open(path, "rb") as f:
            for line in f:
                if line.strip():
                    total += 1
    return total


def load_metadata_counts(path: str) -> Tuple[int, int]:
    """metadata JSON から num_entities / num_relations を読み込む。
    巨大ファイルのフルスキャンは避けるため、本スクリプトはこの値を必須とする。"""
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"--metadata_path '{path}' が存在しません。"
            "巨大ファイルのフルスキャンは避けたいので metadata は必須です。"
            "{\"num_entities\": N, \"num_relations\": M} を含む JSON を渡してください。"
        )
    with open(path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    if "num_entities" not in meta or "num_relations" not in meta:
        raise KeyError(
            f"metadata '{path}' に 'num_entities' または 'num_relations' がありません。"
            f" 受け取ったキー: {list(meta.keys())}"
        )
    return int(meta["num_entities"]), int(meta["num_relations"])


def _iter_file_byte_shard(path: str, start: int, end: int):
    """[start, end) のバイト範囲が割り当てられたシャードがオーナーするバイト行を yield。
    シャードの境界を跨ぐ行は、その行頭バイトを含むシャードが処理する。"""
    with open(path, "rb") as f:
        if start > 0:
            f.seek(start - 1)
            if f.read(1) != b"\n":
                # 直前のシャードが処理する未完の行を読み捨てる
                f.readline()
        # else: 先頭から
        while True:
            pos = f.tell()
            if pos >= end:
                break
            line = f.readline()
            if not line:
                break
            if not line.strip():
                continue
            yield line


class StreamingTripleBatchDataset(IterableDataset):
    """巨大ファイルをワーカープロセスで分担ストリーム読み込みする IterableDataset。
    各ワーカーがファイルをバイト範囲シャーディングして自分の担当だけ読むので、
    多重 I/O コストは抑えつつ Python パース処理を並列化できる。
    バッチをワーカーで構築してテンソル化することで pickle で渡す件数も減らす。"""

    def __init__(
        self,
        paths: List[str],
        batch_size: int,
        input_is_mapped: bool,
        ent2id: Optional[dict],
        rel2id: Optional[dict],
        shuffle_files: bool,
        seed: int,
    ):
        self.paths = list(paths)
        self.batch_size = batch_size
        self.input_is_mapped = input_is_mapped
        self.ent2id = ent2id
        self.rel2id = rel2id
        self.shuffle_files = shuffle_files
        self.seed = seed
        self._iter_count = 0  # __iter__ 毎にインクリメントしてエポック毎に shuffle 種を変える

    def __iter__(self):
        self._iter_count += 1
        info = torch.utils.data.get_worker_info()
        if info is None:
            wid, num = 0, 1
        else:
            wid, num = info.id, info.num_workers

        rng = random.Random(self.seed + self._iter_count * 1000003 + wid * 7919)
        files = self.paths[:]
        if self.shuffle_files:
            rng.shuffle(files)

        bs = self.batch_size
        is_mapped = self.input_is_mapped
        ent2id = self.ent2id
        rel2id = self.rel2id
        buf_h: List[int] = []
        buf_r: List[int] = []
        buf_t: List[int] = []

        for path in files:
            try:
                size = os.path.getsize(path)
            except OSError:
                continue
            if size == 0:
                continue
            shard_start = (size * wid) // num
            shard_end = (size * (wid + 1)) // num
            if shard_start >= shard_end:
                continue

            for line in _iter_file_byte_shard(path, shard_start, shard_end):
                parts = line.split()
                if is_mapped:
                    buf_h.append(int(parts[0]))
                    buf_r.append(int(parts[1]))
                    buf_t.append(int(parts[2]))
                else:
                    buf_h.append(ent2id[parts[0].decode()])
                    buf_r.append(rel2id[parts[1].decode()])
                    buf_t.append(ent2id[parts[2].decode()])

                if len(buf_h) >= bs:
                    yield (
                        torch.tensor(buf_h, dtype=torch.long),
                        torch.tensor(buf_r, dtype=torch.long),
                        torch.tensor(buf_t, dtype=torch.long),
                    )
                    buf_h, buf_r, buf_t = [], [], []

        if buf_h:
            yield (
                torch.tensor(buf_h, dtype=torch.long),
                torch.tensor(buf_r, dtype=torch.long),
                torch.tensor(buf_t, dtype=torch.long),
            )


def make_streaming_loader(
    paths,
    batch_size,
    input_is_mapped,
    ent2id,
    rel2id,
    shuffle_files,
    seed,
    num_workers,
    prefetch_factor,
    persistent_workers,
):
    ds = StreamingTripleBatchDataset(
        paths=paths,
        batch_size=batch_size,
        input_is_mapped=input_is_mapped,
        ent2id=ent2id,
        rel2id=rel2id,
        shuffle_files=shuffle_files,
        seed=seed,
    )
    kwargs = dict(
        batch_size=None,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=(persistent_workers and num_workers > 0),
    )
    if num_workers > 0:
        kwargs["prefetch_factor"] = prefetch_factor
    return DataLoader(ds, **kwargs)


class KGEModel(nn.Module):
    def __init__(self, num_entities, num_relations, emb_dim, sparse_entity=False):
        super().__init__()
        self.entity_emb = nn.Embedding(num_entities, emb_dim, sparse=sparse_entity)
        self.relation_emb = nn.Embedding(num_relations, emb_dim)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.entity_emb.weight)
        nn.init.xavier_uniform_(self.relation_emb.weight)

    def score(self, h_idx, r_idx, t_idx):
        raise NotImplementedError


class TransE(KGEModel):
    def __init__(self, num_entities, num_relations, emb_dim, margin=1.0, p_norm=1, sparse_entity=False):
        super().__init__(num_entities, num_relations, emb_dim, sparse_entity=sparse_entity)
        self.margin = margin
        self.p_norm = p_norm
        with torch.no_grad():
            self.entity_emb.weight.data = F.normalize(self.entity_emb.weight.data, p=2, dim=1)
            self.relation_emb.weight.data = F.normalize(self.relation_emb.weight.data, p=2, dim=1)

    def score(self, h_idx, r_idx, t_idx):
        h = self.entity_emb(h_idx)
        r = self.relation_emb(r_idx)
        t = self.entity_emb(t_idx)
        return -torch.norm(h + r - t, p=self.p_norm, dim=-1)


class DistMult(KGEModel):
    def score(self, h_idx, r_idx, t_idx):
        h = self.entity_emb(h_idx)
        r = self.relation_emb(r_idx)
        t = self.entity_emb(t_idx)
        return torch.sum(h * r * t, dim=-1)


def build_model(name, num_entities, num_relations, emb_dim, margin, sparse_entity):
    name = name.lower()
    if name == "transe":
        return TransE(num_entities, num_relations, emb_dim, margin=margin, p_norm=1,
                      sparse_entity=sparse_entity)
    if name == "distmult":
        return DistMult(num_entities, num_relations, emb_dim, sparse_entity=sparse_entity)
    raise ValueError(f"Unknown model: {name}")


def build_optimizers(model, lr, sparse_entity):
    """sparse=True の embedding は SparseAdam、それ以外は Adam に分けて並走させる。"""
    if not sparse_entity:
        return [torch.optim.Adam(model.parameters(), lr=lr)]
    sparse_params = [p for n, p in model.named_parameters() if n == "entity_emb.weight"]
    dense_params = [p for n, p in model.named_parameters() if n != "entity_emb.weight"]
    opts = []
    if sparse_params:
        opts.append(torch.optim.SparseAdam(sparse_params, lr=lr))
    if dense_params:
        opts.append(torch.optim.Adam(dense_params, lr=lr))
    return opts


def train_one_epoch(
    model,
    optimizers,
    train_loader,
    num_entities,
    num_negatives,
    margin,
    device,
    use_amp,
    scaler,
    max_batches=0,
    log_every=0,
):
    model.train()
    total_loss = torch.zeros((), device=device)
    total_examples = 0
    is_transe = isinstance(model, TransE)

    t_loop = time.time()
    for batch_idx, (h, r, t) in enumerate(train_loader):
        if max_batches > 0 and batch_idx >= max_batches:
            break
        h = h.to(device, non_blocking=True)
        r = r.to(device, non_blocking=True)
        t = t.to(device, non_blocking=True)
        B = h.size(0)
        K = num_negatives

        for opt in optimizers:
            opt.zero_grad(set_to_none=True)

        with torch.amp.autocast(device_type="cuda", enabled=use_amp):
            pos_score = model.score(h, r, t)

            neg_e = torch.randint(0, num_entities, (B, K), device=device)
            corrupt_head = torch.rand(B, K, device=device) < 0.5

            h_rep = h.unsqueeze(1).expand(B, K)
            r_rep = r.unsqueeze(1).expand(B, K)
            t_rep = t.unsqueeze(1).expand(B, K)

            neg_h = torch.where(corrupt_head, neg_e, h_rep)
            neg_t = torch.where(corrupt_head, t_rep, neg_e)

            neg_score = model.score(
                neg_h.reshape(-1),
                r_rep.reshape(-1),
                neg_t.reshape(-1),
            ).view(B, K)

            loss = F.relu(margin - pos_score.unsqueeze(1) + neg_score).mean()

        if use_amp and scaler is not None:
            scaler.scale(loss).backward()
            for opt in optimizers:
                scaler.step(opt)
            scaler.update()
        else:
            loss.backward()
            for opt in optimizers:
                opt.step()

        if is_transe:
            with torch.no_grad():
                model.entity_emb.weight.data = F.normalize(model.entity_emb.weight.data, p=2, dim=1)
                model.relation_emb.weight.data = F.normalize(model.relation_emb.weight.data, p=2, dim=1)

        total_loss = total_loss + loss.detach() * B
        total_examples += B

        if log_every > 0 and (batch_idx + 1) % log_every == 0:
            torch.cuda.synchronize() if device.type == "cuda" else None
            print(
                f"  step {batch_idx + 1}: loss={loss.item():.4f} "
                f"elapsed={time.time() - t_loop:.1f}s",
                flush=True,
            )

    return (total_loss / max(total_examples, 1)).item()


def _chunk_scores_tail(model, x, cand):
    """x: [B, D]  cand: [C, D]  -> scores [B, C]"""
    if isinstance(model, DistMult):
        return x @ cand.t()
    p_norm = getattr(model, "p_norm", 1)
    diff = x.unsqueeze(1) - cand.unsqueeze(0)  # [B, C, D]
    if p_norm == 1:
        return -diff.abs().sum(dim=-1)
    if p_norm == 2:
        return -torch.sqrt(torch.clamp((diff * diff).sum(dim=-1), min=1e-12))
    return -torch.norm(diff, p=p_norm, dim=-1)


@torch.no_grad()
def _compute_tail_ranks(model, h, r, t, num_entities, entity_chunk_size):
    """[B, num_entities] を materialize せずチャンク毎にランクを集計する。
    巨大 num_entities (>1e8 など) でも OOM しない。"""
    device = h.device
    B = h.size(0)
    true_scores = model.score(h, r, t)  # [B]

    if isinstance(model, DistMult):
        x = model.entity_emb(h) * model.relation_emb(r)
    else:
        x = model.entity_emb(h) + model.relation_emb(r)

    ranks = torch.ones(B, dtype=torch.long, device=device)
    for s in range(0, num_entities, entity_chunk_size):
        e = min(s + entity_chunk_size, num_entities)
        cand = model.entity_emb.weight[s:e]
        scores = _chunk_scores_tail(model, x, cand)
        ranks += (scores > true_scores.unsqueeze(1)).sum(dim=1)
    return ranks


@torch.no_grad()
def _compute_head_ranks(model, h, r, t, num_entities, entity_chunk_size):
    device = h.device
    B = h.size(0)
    true_scores = model.score(h, r, t)

    if isinstance(model, DistMult):
        x = model.relation_emb(r) * model.entity_emb(t)
    else:
        x = model.entity_emb(t) - model.relation_emb(r)

    ranks = torch.ones(B, dtype=torch.long, device=device)
    for s in range(0, num_entities, entity_chunk_size):
        e = min(s + entity_chunk_size, num_entities)
        cand = model.entity_emb.weight[s:e]
        scores = _chunk_scores_tail(model, x, cand)
        ranks += (scores > true_scores.unsqueeze(1)).sum(dim=1)
    return ranks


@torch.no_grad()
def evaluate_unfiltered(
    model,
    eval_loader,
    num_entities,
    device,
    entity_chunk_size,
    max_batches=0,
):
    model.eval()
    rank_chunks = []

    for batch_idx, (h, r, t) in enumerate(eval_loader):
        if max_batches > 0 and batch_idx >= max_batches:
            break
        h = h.to(device, non_blocking=True)
        r = r.to(device, non_blocking=True)
        t = t.to(device, non_blocking=True)

        rank_chunks.append(_compute_tail_ranks(model, h, r, t, num_entities, entity_chunk_size))
        rank_chunks.append(_compute_head_ranks(model, h, r, t, num_entities, entity_chunk_size))

    if not rank_chunks:
        return {"mrr": 0.0, "hits1": 0.0, "hits3": 0.0, "hits10": 0.0}

    ranks = torch.cat(rank_chunks).float()
    mrr = torch.mean(1.0 / ranks).item()
    hits1 = torch.mean((ranks <= 1).float()).item()
    hits3 = torch.mean((ranks <= 3).float()).item()
    hits10 = torch.mean((ranks <= 10).float()).item()
    return {"mrr": mrr, "hits1": hits1, "hits3": hits3, "hits10": hits10}


def _save_one_embedding(name, weight_gpu, npy_path, tsv_head_path, id2raw, preview_rows):
    """埋め込み行列を 1 つ保存する。
    - 全件を `.npy` バイナリで(numpy.save)
    - 先頭 N 行のみ `.tsv` で人が読めるプレビュー出力"""
    print(f"Saving {name} embeddings (shape={tuple(weight_gpu.shape)}) ...", flush=True)
    t0 = time.time()
    weight_cpu = weight_gpu.detach().to("cpu", copy=True)
    arr = weight_cpu.numpy()
    np.save(npy_path, arr)
    print(f"  npy : {npy_path}  ({arr.nbytes / (1024**3):.2f} GB, {time.time() - t0:.1f}s)",
          flush=True)

    n_preview = min(preview_rows, weight_cpu.size(0))
    with open(tsv_head_path, "w", encoding="utf-8") as f:
        f.write("raw_id\tinternal_id\tembedding\n")
        for i in range(n_preview):
            raw_id = id2raw[i] if id2raw is not None else str(i)
            vec = " ".join(f"{x:.8f}" for x in weight_cpu[i].tolist())
            f.write(f"{raw_id}\t{i}\t{vec}\n")
    print(f"  tsv preview ({n_preview} rows): {tsv_head_path}", flush=True)


def save_embeddings(output_dir, model, entity_mapping=None, relation_mapping=None,
                    preview_rows=32):
    os.makedirs(output_dir, exist_ok=True)

    id2ent = None
    id2rel = None
    if entity_mapping is not None:
        id2ent = {v: k for k, v in entity_mapping.items()}
    if relation_mapping is not None:
        id2rel = {v: k for k, v in relation_mapping.items()}

    _save_one_embedding(
        name="entity",
        weight_gpu=model.entity_emb.weight,
        npy_path=os.path.join(output_dir, "entity_embeddings.npy"),
        tsv_head_path=os.path.join(output_dir, "entity_embeddings_head.tsv"),
        id2raw=id2ent,
        preview_rows=preview_rows,
    )
    _save_one_embedding(
        name="relation",
        weight_gpu=model.relation_emb.weight,
        npy_path=os.path.join(output_dir, "relation_embeddings.npy"),
        tsv_head_path=os.path.join(output_dir, "relation_embeddings_head.tsv"),
        id2raw=id2rel,
        preview_rows=preview_rows,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_paths", nargs="+", required=True)
    parser.add_argument("--valid_paths", nargs="+", required=True)
    parser.add_argument("--test_paths", nargs="+", required=True)

    parser.add_argument("--input_is_mapped", type=str2bool, default=True)
    parser.add_argument("--entity_mapping_path", type=str, default=None)
    parser.add_argument("--relation_mapping_path", type=str, default=None)
    parser.add_argument("--metadata_path", type=str, default=None)

    parser.add_argument("--model", type=str, default="transe", choices=["transe", "distmult"])
    parser.add_argument("--emb_dim", type=int, default=16)
    parser.add_argument("--batch_size", type=int, default=4096)
    parser.add_argument("--num_negatives", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--margin", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output_dir", type=str, default="./outputs")

    parser.add_argument("--num_workers", type=int, default=4,
                        help="ストリーム I/O 並列数。巨大ファイルなら 4-8 程度を推奨")
    parser.add_argument("--prefetch_factor", type=int, default=4)
    parser.add_argument("--use_amp", type=str2bool, default=True)
    parser.add_argument("--sparse_entity", type=str2bool, default=False,
                        help="True にすると entity_emb を sparse 化し SparseAdam で更新する。"
                             "AMP は自動的に無効化される")
    parser.add_argument("--eval_every", type=int, default=1)
    parser.add_argument("--eval_batch_size", type=int, default=256)
    parser.add_argument("--entity_chunk_size", type=int, default=20000)
    parser.add_argument("--skip_count_lines", type=str2bool, default=True,
                        help="巨大ファイルでの行数カウントをスキップ(デフォルト True)")
    parser.add_argument("--max_train_batches", type=int, default=0,
                        help="0=無制限。動作確認やベンチマーク用に1エポック内のバッチ数を制限")
    parser.add_argument("--max_eval_batches", type=int, default=0,
                        help="0=無制限。動作確認やベンチマーク用に評価バッチ数を制限")
    parser.add_argument("--log_every", type=int, default=0,
                        help="0=無効。N ステップ毎に loss と経過時間を出力")
    parser.add_argument("--skip_save_embeddings", type=str2bool, default=False,
                        help="埋め込み出力を完全にスキップ")
    parser.add_argument("--tsv_preview_rows", type=int, default=32,
                        help="人が見るための TSV プレビュー行数(全件は .npy で出力)")
    parser.add_argument("--save_best_state_on_cpu", type=str2bool, default=True,
                        help="False にすると最良エポックの state を GPU 上に保持(巨大埋め込みの GPU↔CPU 往復を回避)")

    args = parser.parse_args()

    set_seed(args.seed)

    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    ent2id = None
    rel2id = None

    if not args.input_is_mapped:
        if args.entity_mapping_path is None or args.relation_mapping_path is None:
            raise ValueError("input_is_mapped=false のとき mapping ファイルが必要です。")
        ent2id = load_mapping(args.entity_mapping_path)
        rel2id = load_mapping(args.relation_mapping_path)
        num_entities = len(ent2id)
        num_relations = len(rel2id)
    else:
        if args.metadata_path is None:
            raise ValueError(
                "input_is_mapped=true のときは --metadata_path が必須です "
                "(巨大ファイルの全スキャンを避けるため)。"
                "{\"num_entities\": N, \"num_relations\": M} を含む JSON を渡してください。"
            )
        num_entities, num_relations = load_metadata_counts(args.metadata_path)
        print("#entities:", num_entities)
        print("#relations:", num_relations)

    device = torch.device(args.device)
    use_amp = args.use_amp and device.type == "cuda" and not args.sparse_entity

    # 巨大エンティティの埋め込みを CPU に確保→転送する経路は重いので、最初から device 上で構築する。
    print(f"Constructing model on {device} (num_entities={num_entities}, emb_dim={args.emb_dim}) ...",
          flush=True)
    t_build = time.time()
    with torch.device(device):
        model = build_model(
            args.model, num_entities, num_relations, args.emb_dim, args.margin,
            sparse_entity=args.sparse_entity,
        )
    if device.type == "cuda":
        torch.cuda.synchronize()
    print(f"  built in {time.time() - t_build:.1f}s", flush=True)
    optimizers = build_optimizers(model, args.lr, args.sparse_entity)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    print("=== Dataset summary ===")
    if not args.skip_count_lines:
        print(f"#train triples: {count_lines(args.train_paths)}")
        print(f"#valid triples: {count_lines(args.valid_paths)}")
        print(f"#test triples : {count_lines(args.test_paths)}")
    print(f"#entities     : {num_entities}")
    print(f"#relations    : {num_relations}")
    print(
        f"batch={args.batch_size}, negs={args.num_negatives}, "
        f"workers={args.num_workers}, amp={use_amp}, "
        f"sparse_entity={args.sparse_entity}",
        flush=True,
    )

    train_loader = make_streaming_loader(
        paths=args.train_paths,
        batch_size=args.batch_size,
        input_is_mapped=args.input_is_mapped,
        ent2id=ent2id,
        rel2id=rel2id,
        shuffle_files=True,
        seed=args.seed,
        num_workers=args.num_workers,
        prefetch_factor=args.prefetch_factor,
        persistent_workers=True,
    )
    valid_loader = make_streaming_loader(
        paths=args.valid_paths,
        batch_size=args.eval_batch_size,
        input_is_mapped=args.input_is_mapped,
        ent2id=ent2id,
        rel2id=rel2id,
        shuffle_files=False,
        seed=args.seed,
        num_workers=max(1, args.num_workers // 2),
        prefetch_factor=args.prefetch_factor,
        persistent_workers=True,
    )

    best_valid_mrr = -1.0
    best_state = None

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss = train_one_epoch(
            model=model,
            optimizers=optimizers,
            train_loader=train_loader,
            num_entities=num_entities,
            num_negatives=args.num_negatives,
            margin=args.margin,
            device=device,
            use_amp=use_amp,
            scaler=scaler,
            max_batches=args.max_train_batches,
            log_every=args.log_every,
        )
        train_dt = time.time() - t0

        do_eval = (epoch % args.eval_every == 0) or (epoch == args.epochs)
        if do_eval:
            t1 = time.time()
            valid_result = evaluate_unfiltered(
                model=model,
                eval_loader=valid_loader,
                num_entities=num_entities,
                device=device,
                entity_chunk_size=args.entity_chunk_size,
                max_batches=args.max_eval_batches,
            )
            eval_dt = time.time() - t1
            print(
                f"[Epoch {epoch:03d}] "
                f"loss={train_loss:.4f} | "
                f"valid MRR={valid_result['mrr']:.4f} "
                f"H@1={valid_result['hits1']:.4f} "
                f"H@3={valid_result['hits3']:.4f} "
                f"H@10={valid_result['hits10']:.4f} | "
                f"train {train_dt:.1f}s eval {eval_dt:.1f}s",
                flush=True,
            )
            if valid_result["mrr"] > best_valid_mrr:
                best_valid_mrr = valid_result["mrr"]
                if args.save_best_state_on_cpu:
                    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                else:
                    best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            print(
                f"[Epoch {epoch:03d}] loss={train_loss:.4f} | train {train_dt:.1f}s (eval skipped)",
                flush=True,
            )

    if best_state is not None:
        model.load_state_dict(best_state)

    test_loader = make_streaming_loader(
        paths=args.test_paths,
        batch_size=args.eval_batch_size,
        input_is_mapped=args.input_is_mapped,
        ent2id=ent2id,
        rel2id=rel2id,
        shuffle_files=False,
        seed=args.seed,
        num_workers=max(1, args.num_workers // 2),
        prefetch_factor=args.prefetch_factor,
        persistent_workers=False,
    )
    test_result = evaluate_unfiltered(
        model=model,
        eval_loader=test_loader,
        num_entities=num_entities,
        device=device,
        entity_chunk_size=args.entity_chunk_size,
        max_batches=args.max_eval_batches,
    )

    print("\n=== Final Test Evaluation (unfiltered) ===")
    print(f"Test MRR    : {test_result['mrr']:.4f}")
    print(f"Test Hits@1 : {test_result['hits1']:.4f}")
    print(f"Test Hits@3 : {test_result['hits3']:.4f}")
    print(f"Test Hits@10: {test_result['hits10']:.4f}")

    if args.skip_save_embeddings:
        print("(skip_save_embeddings=True: 埋め込み出力をスキップ)", flush=True)
    else:
        save_embeddings(
            output_dir=args.output_dir,
            model=model,
            entity_mapping=ent2id,
            relation_mapping=rel2id,
            preview_rows=args.tsv_preview_rows,
        )


if __name__ == "__main__":
    main()
