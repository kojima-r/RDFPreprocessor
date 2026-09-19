import argparse
import os
import random
import time
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def read_tsv_triples(path: str):
    """
    TSV形式:
        head<TAB>relation<TAB>tail
    """
    triples = []
    raw_entities = set()
    raw_relations = set()

    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) != 3:
                raise ValueError(f"Line {line_no}: expected 3 columns, got {len(parts)} -> {line}")
            h_raw, r_raw, t_raw = parts
            raw_entities.add(h_raw)
            raw_entities.add(t_raw)
            raw_relations.add(r_raw)
            triples.append((h_raw, r_raw, t_raw))

    ent2id = {e: i for i, e in enumerate(sorted(raw_entities))}
    rel2id = {r: i for i, r in enumerate(sorted(raw_relations))}
    id2ent = {i: e for e, i in ent2id.items()}
    id2rel = {i: r for r, i in rel2id.items()}

    mapped_triples = [(ent2id[h], rel2id[r], ent2id[t]) for h, r, t in triples]
    return mapped_triples, ent2id, rel2id, id2ent, id2rel


def split_triples(triples, train_ratio=0.8, valid_ratio=0.1, seed=42):
    triples = triples[:]
    rng = random.Random(seed)
    rng.shuffle(triples)
    n = len(triples)
    n_train = int(n * train_ratio)
    n_valid = int(n * valid_ratio)
    train = triples[:n_train]
    valid = triples[n_train:n_train + n_valid]
    test = triples[n_train + n_valid:]
    return train, valid, test


def triples_to_tensors(triples, device):
    """三つ組リストを GPU 上の LongTensor 3本に一括転送する。"""
    arr = torch.tensor(triples, dtype=torch.long)
    arr = arr.to(device, non_blocking=True)
    return arr[:, 0].contiguous(), arr[:, 1].contiguous(), arr[:, 2].contiguous()


class KGEModel(nn.Module):
    def __init__(self, num_entities, num_relations, emb_dim):
        super().__init__()
        self.num_entities = num_entities
        self.num_relations = num_relations
        self.emb_dim = emb_dim
        self.entity_emb = nn.Embedding(num_entities, emb_dim)
        self.relation_emb = nn.Embedding(num_relations, emb_dim)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.entity_emb.weight)
        nn.init.xavier_uniform_(self.relation_emb.weight)

    def score(self, h_idx, r_idx, t_idx):
        raise NotImplementedError


class TransE(KGEModel):
    def __init__(self, num_entities, num_relations, emb_dim, margin=1.0, p_norm=1):
        super().__init__(num_entities, num_relations, emb_dim)
        self.margin = margin
        self.p_norm = p_norm

    def reset_parameters(self):
        super().reset_parameters()
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


def train_one_epoch(
    model,
    train_h, train_r, train_t,
    optimizer,
    num_entities,
    batch_size,
    num_negatives,
    margin,
    use_amp,
    scaler,
):
    """全 triple は GPU 上に常駐させ、permutation でミニバッチ化する。
    複数 negative をベクトル化して採点することで forward あたりの GPU 計算量を増やす。"""
    model.train()
    device = train_h.device
    N = train_h.size(0)
    perm = torch.randperm(N, device=device)

    total_loss = torch.zeros((), device=device)
    total_examples = 0
    is_transe = isinstance(model, TransE)

    for start in range(0, N, batch_size):
        idx = perm[start:start + batch_size]
        h = train_h[idx]
        r = train_r[idx]
        t = train_t[idx]
        B = h.size(0)
        K = num_negatives

        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast(device_type="cuda", enabled=use_amp):
            pos_score = model.score(h, r, t)  # [B]

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

        if scaler is not None and use_amp:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        if is_transe:
            with torch.no_grad():
                model.entity_emb.weight.data = F.normalize(model.entity_emb.weight.data, p=2, dim=1)
                model.relation_emb.weight.data = F.normalize(model.relation_emb.weight.data, p=2, dim=1)

        total_loss = total_loss + loss.detach() * B
        total_examples += B

    return (total_loss / max(total_examples, 1)).item()


@dataclass
class EvalResult:
    mrr: float
    hits1: float
    hits3: float
    hits10: float


def build_filter_dict(all_triples):
    hr_to_t = defaultdict(list)
    rt_to_h = defaultdict(list)
    for h, r, t in all_triples:
        hr_to_t[(h, r)].append(t)
        rt_to_h[(r, t)].append(h)
    return hr_to_t, rt_to_h


@torch.no_grad()
def _score_all_tails(model, h, r, num_entities, entity_chunk_size):
    """[B, E] 全 tail スコア。DistMult はそのまま matmul、TransE は chunk 化。"""
    device = h.device
    B = h.size(0)

    if isinstance(model, DistMult):
        x = model.entity_emb(h) * model.relation_emb(r)            # [B, D]
        return x @ model.entity_emb.weight.t()                     # [B, E]

    x = model.entity_emb(h) + model.relation_emb(r)                # [B, D]
    p_norm = getattr(model, "p_norm", 1)
    out = torch.empty((B, num_entities), device=device, dtype=x.dtype)
    for s in range(0, num_entities, entity_chunk_size):
        e = min(s + entity_chunk_size, num_entities)
        cand = model.entity_emb.weight[s:e]                        # [C, D]
        diff = x.unsqueeze(1) - cand.unsqueeze(0)                  # [B, C, D]
        if p_norm == 1:
            out[:, s:e] = -diff.abs().sum(dim=-1)
        elif p_norm == 2:
            out[:, s:e] = -torch.sqrt(torch.clamp((diff * diff).sum(dim=-1), min=1e-12))
        else:
            out[:, s:e] = -torch.norm(diff, p=p_norm, dim=-1)
    return out


@torch.no_grad()
def _score_all_heads(model, r, t, num_entities, entity_chunk_size):
    device = r.device
    B = r.size(0)

    if isinstance(model, DistMult):
        x = model.relation_emb(r) * model.entity_emb(t)
        return x @ model.entity_emb.weight.t()

    x = model.entity_emb(t) - model.relation_emb(r)
    p_norm = getattr(model, "p_norm", 1)
    out = torch.empty((B, num_entities), device=device, dtype=x.dtype)
    for s in range(0, num_entities, entity_chunk_size):
        e = min(s + entity_chunk_size, num_entities)
        cand = model.entity_emb.weight[s:e]
        diff = x.unsqueeze(1) - cand.unsqueeze(0)
        if p_norm == 1:
            out[:, s:e] = -diff.abs().sum(dim=-1)
        elif p_norm == 2:
            out[:, s:e] = -torch.sqrt(torch.clamp((diff * diff).sum(dim=-1), min=1e-12))
        else:
            out[:, s:e] = -torch.norm(diff, p=p_norm, dim=-1)
    return out


def _build_filter_pad(filter_lists: List[List[int]], default_ids: List[int], device):
    """[B, max_len] のパディング tensor を CPU で構築し一度だけ GPU に転送。
    パディング箇所には true entity id を入れておくので、その後の真スコア再書き込みで上書きされ無害化される。"""
    B = len(filter_lists)
    max_len = max((len(x) for x in filter_lists), default=0)
    if max_len == 0:
        return None
    pad_cpu = torch.empty((B, max_len), dtype=torch.long)
    for i, ents in enumerate(filter_lists):
        L = len(ents)
        if L > 0:
            pad_cpu[i, :L] = torch.as_tensor(ents, dtype=torch.long)
        if L < max_len:
            pad_cpu[i, L:] = default_ids[i]
    return pad_cpu.to(device, non_blocking=True)


@torch.no_grad()
def evaluate(
    model,
    triples,
    device,
    hr_to_t,
    rt_to_h,
    batch_size=128,
    filtered=True,
    entity_chunk_size=20000,
):
    """フィルタリング MRR/Hits@k。
    旧実装が「chunk × per-row Python ループ」で score を書き換えていたのを、
    `[B, E]` を一度だけ作って `scatter_` で一括マスクするように変更。"""
    model.eval()
    num_entities = model.entity_emb.num_embeddings
    NEG_INF = -1e9

    rank_chunks = []

    for start in range(0, len(triples), batch_size):
        batch = triples[start:start + batch_size]
        arr = torch.tensor(batch, dtype=torch.long, device=device)
        h = arr[:, 0]
        r = arr[:, 1]
        t = arr[:, 2]

        h_list = h.tolist()
        r_list = r.tolist()
        t_list = t.tolist()

        # ---------- tail prediction ----------
        scores = _score_all_tails(model, h, r, num_entities, entity_chunk_size)
        true_scores = scores.gather(1, t.unsqueeze(1)).squeeze(1)
        if filtered:
            pad = _build_filter_pad(
                [hr_to_t[(hi, ri)] for hi, ri in zip(h_list, r_list)],
                t_list,
                device,
            )
            if pad is not None:
                scores.scatter_(1, pad, NEG_INF)
                scores.scatter_(1, t.unsqueeze(1), true_scores.unsqueeze(1))
        rank_chunks.append((scores > true_scores.unsqueeze(1)).sum(dim=1) + 1)

        # ---------- head prediction ----------
        scores = _score_all_heads(model, r, t, num_entities, entity_chunk_size)
        true_scores = scores.gather(1, h.unsqueeze(1)).squeeze(1)
        if filtered:
            pad = _build_filter_pad(
                [rt_to_h[(ri, ti)] for ri, ti in zip(r_list, t_list)],
                h_list,
                device,
            )
            if pad is not None:
                scores.scatter_(1, pad, NEG_INF)
                scores.scatter_(1, h.unsqueeze(1), true_scores.unsqueeze(1))
        rank_chunks.append((scores > true_scores.unsqueeze(1)).sum(dim=1) + 1)

    ranks = torch.cat(rank_chunks).float()
    mrr = torch.mean(1.0 / ranks).item()
    hits1 = torch.mean((ranks <= 1).float()).item()
    hits3 = torch.mean((ranks <= 3).float()).item()
    hits10 = torch.mean((ranks <= 10).float()).item()
    return EvalResult(mrr=mrr, hits1=hits1, hits3=hits3, hits10=hits10)


def build_model(name, num_entities, num_relations, emb_dim, margin):
    name = name.lower()
    if name == "transe":
        return TransE(num_entities, num_relations, emb_dim, margin=margin, p_norm=1)
    elif name == "distmult":
        return DistMult(num_entities, num_relations, emb_dim)
    else:
        raise ValueError(f"Unknown model: {name}")


def save_embeddings(output_dir, model, id2ent, id2rel):
    os.makedirs(output_dir, exist_ok=True)
    entity_path = os.path.join(output_dir, "entity_embeddings.tsv")
    relation_path = os.path.join(output_dir, "relation_embeddings.tsv")

    entity_weights = model.entity_emb.weight.detach().cpu()
    relation_weights = model.relation_emb.weight.detach().cpu()

    with open(entity_path, "w", encoding="utf-8") as f:
        f.write("raw_id\tinternal_id\tembedding\n")
        for internal_id in range(entity_weights.size(0)):
            raw_id = id2ent[internal_id]
            vec = entity_weights[internal_id].tolist()
            vec_str = " ".join(f"{x:.8f}" for x in vec)
            f.write(f"{raw_id}\t{internal_id}\t{vec_str}\n")

    with open(relation_path, "w", encoding="utf-8") as f:
        f.write("raw_id\tinternal_id\tembedding\n")
        for internal_id in range(relation_weights.size(0)):
            raw_id = id2rel[internal_id]
            vec = relation_weights[internal_id].tolist()
            vec_str = " ".join(f"{x:.8f}" for x in vec)
            f.write(f"{raw_id}\t{internal_id}\t{vec_str}\n")

    print(f"Saved entity embeddings to: {entity_path}")
    print(f"Saved relation embeddings to: {relation_path}")


def str2bool(v):
    if isinstance(v, bool):
        return v
    v = v.lower()
    if v in ("true", "1", "yes", "y"):
        return True
    if v in ("false", "0", "no", "n"):
        return False
    raise ValueError(f"invalid bool: {v}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True, help="TSV triple file path")
    parser.add_argument("--model", type=str, default="transe", choices=["transe", "distmult"])
    parser.add_argument("--emb_dim", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=4096)
    parser.add_argument("--num_negatives", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--margin", type=float, default=1.0)
    parser.add_argument("--train_ratio", type=float, default=0.8)
    parser.add_argument("--valid_ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str, default="./output_kg")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--filtered_eval", type=str2bool, default=True)
    parser.add_argument("--eval_batch_size", type=int, default=128)
    parser.add_argument("--entity_chunk_size", type=int, default=20000)
    parser.add_argument("--eval_every", type=int, default=5, help="N epoch ごとに valid 評価")
    parser.add_argument("--use_amp", type=str2bool, default=True)
    args = parser.parse_args()

    set_seed(args.seed)

    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    triples, ent2id, rel2id, id2ent, id2rel = read_tsv_triples(args.data_path)
    train_triples, valid_triples, test_triples = split_triples(
        triples,
        train_ratio=args.train_ratio,
        valid_ratio=args.valid_ratio,
        seed=args.seed,
    )

    num_entities = len(ent2id)
    num_relations = len(rel2id)

    print("=== Dataset summary ===")
    print(f"#triples   : {len(triples)}")
    print(f"#entities  : {num_entities}")
    print(f"#relations : {num_relations}")
    print(f"#train     : {len(train_triples)}")
    print(f"#valid     : {len(valid_triples)}")
    print(f"#test      : {len(test_triples)}")
    print()

    device = torch.device(args.device)
    use_amp = args.use_amp and device.type == "cuda"

    train_h, train_r, train_t = triples_to_tensors(train_triples, device)

    model = build_model(args.model, num_entities, num_relations, args.emb_dim, args.margin).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    if args.filtered_eval:
        print("Building filter dict ...", flush=True)
        all_known_triples = train_triples + valid_triples + test_triples
        hr_to_t, rt_to_h = build_filter_dict(all_known_triples)
    else:
        hr_to_t, rt_to_h = None, None

    best_valid_mrr = -1.0
    best_state = None

    print(
        f"=== Training {args.model} "
        f"(batch={args.batch_size}, negs={args.num_negatives}, "
        f"amp={use_amp}, eval_every={args.eval_every}) ===",
        flush=True,
    )

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss = train_one_epoch(
            model=model,
            train_h=train_h, train_r=train_r, train_t=train_t,
            optimizer=optimizer,
            num_entities=num_entities,
            batch_size=args.batch_size,
            num_negatives=args.num_negatives,
            margin=args.margin,
            use_amp=use_amp,
            scaler=scaler,
        )
        train_dt = time.time() - t0

        do_eval = (epoch % args.eval_every == 0) or (epoch == args.epochs)
        if do_eval:
            t1 = time.time()
            valid_result = evaluate(
                model=model,
                triples=valid_triples,
                device=device,
                hr_to_t=hr_to_t,
                rt_to_h=rt_to_h,
                batch_size=args.eval_batch_size,
                filtered=args.filtered_eval,
                entity_chunk_size=args.entity_chunk_size,
            )
            eval_dt = time.time() - t1
            print(
                f"[Epoch {epoch:03d}] "
                f"loss={train_loss:.4f} | "
                f"valid MRR={valid_result.mrr:.4f} "
                f"H@1={valid_result.hits1:.4f} "
                f"H@3={valid_result.hits3:.4f} "
                f"H@10={valid_result.hits10:.4f} | "
                f"train {train_dt:.1f}s eval {eval_dt:.1f}s",
                flush=True,
            )
            if valid_result.mrr > best_valid_mrr:
                best_valid_mrr = valid_result.mrr
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            print(
                f"[Epoch {epoch:03d}] loss={train_loss:.4f} | train {train_dt:.1f}s (eval skipped)",
                flush=True,
            )

    if best_state is not None:
        model.load_state_dict(best_state)

    print("\n=== Final Test Evaluation ===")
    test_result = evaluate(
        model=model,
        triples=test_triples,
        device=device,
        hr_to_t=hr_to_t,
        rt_to_h=rt_to_h,
        batch_size=args.eval_batch_size,
        filtered=args.filtered_eval,
        entity_chunk_size=args.entity_chunk_size,
    )

    print(f"Test MRR    : {test_result.mrr:.4f}")
    print(f"Test Hits@1 : {test_result.hits1:.4f}")
    print(f"Test Hits@3 : {test_result.hits3:.4f}")
    print(f"Test Hits@10: {test_result.hits10:.4f}")

    save_embeddings(
        output_dir=args.output_dir,
        model=model,
        id2ent=id2ent,
        id2rel=id2rel,
    )


if __name__ == "__main__":
    main()
