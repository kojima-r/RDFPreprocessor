import argparse
import os
import random
from collections import defaultdict
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Tuple, Optional

def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def read_tsv_triples(path: str):
    """
    TSV形式:
        head<TAB>relation<TAB>tail
    例:
        521686\t7\t724659087
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


class KGDataset(Dataset):
    def __init__(self, triples):
        self.triples = triples

    def __len__(self):
        return len(self.triples)

    def __getitem__(self, idx):
        return self.triples[idx]


def collate_fn(batch):
    h = torch.tensor([x[0] for x in batch], dtype=torch.long)
    r = torch.tensor([x[1] for x in batch], dtype=torch.long)
    t = torch.tensor([x[2] for x in batch], dtype=torch.long)
    return h, r, t


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

    def score_all_tails(self, h_idx, r_idx):
        raise NotImplementedError

    def score_all_heads(self, r_idx, t_idx):
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

    def score_all_tails(self, h_idx, r_idx):
        h = self.entity_emb(h_idx)
        r = self.relation_emb(r_idx)
        all_t = self.entity_emb.weight
        x = h + r
        return -torch.cdist(x, all_t, p=self.p_norm)

    def score_all_heads(self, r_idx, t_idx):
        r = self.relation_emb(r_idx)
        t = self.entity_emb(t_idx)
        all_h = self.entity_emb.weight
        x = t - r
        return -torch.cdist(x, all_h, p=self.p_norm)


class DistMult(KGEModel):
    def score(self, h_idx, r_idx, t_idx):
        h = self.entity_emb(h_idx)
        r = self.relation_emb(r_idx)
        t = self.entity_emb(t_idx)
        return torch.sum(h * r * t, dim=-1)

    def score_all_tails(self, h_idx, r_idx):
        h = self.entity_emb(h_idx)
        r = self.relation_emb(r_idx)
        x = h * r
        all_t = self.entity_emb.weight
        return x @ all_t.t()

    def score_all_heads(self, r_idx, t_idx):
        r = self.relation_emb(r_idx)
        t = self.entity_emb(t_idx)
        x = r * t
        all_h = self.entity_emb.weight
        return x @ all_h.t()


def sample_negative_tails(pos_tails, num_entities, device):
    return torch.randint(0, num_entities, size=pos_tails.shape, device=device)


def sample_negative_heads(pos_heads, num_entities, device):
    return torch.randint(0, num_entities, size=pos_heads.shape, device=device)


def train_one_epoch(model, loader, optimizer, device, num_entities, margin=1.0):
    model.train()
    total_loss = 0.0
    total_examples = 0

    for h, r, t in loader:
        h = h.to(device)
        r = r.to(device)
        t = t.to(device)

        optimizer.zero_grad()

        corrupt_head = torch.rand(h.size(0), device=device) < 0.5

        neg_h = h.clone()
        neg_t = t.clone()

        if corrupt_head.any():
            neg_h[corrupt_head] = sample_negative_heads(h[corrupt_head], num_entities, device)
        if (~corrupt_head).any():
            neg_t[~corrupt_head] = sample_negative_tails(t[~corrupt_head], num_entities, device)

        pos_score = model.score(h, r, t)
        neg_score = model.score(neg_h, r, neg_t)

        loss = F.relu(margin - pos_score + neg_score).mean()
        loss.backward()
        optimizer.step()

        if isinstance(model, TransE):
            with torch.no_grad():
                model.entity_emb.weight.data = F.normalize(model.entity_emb.weight.data, p=2, dim=1)
                model.relation_emb.weight.data = F.normalize(model.relation_emb.weight.data, p=2, dim=1)

        batch_size = h.size(0)
        total_loss += loss.item() * batch_size
        total_examples += batch_size

    return total_loss / max(total_examples, 1)


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

def _make_padded_filter_index(
    keys: List[Tuple[int, int]],
    filter_dict: Dict[Tuple[int, int], List[int]],
    device: torch.device,
):
    lists = [filter_dict[k] for k in keys]
    max_len = max((len(x) for x in lists), default=0)

    if max_len == 0:
        padded_idx = torch.empty((len(keys), 0), dtype=torch.long, device=device)
        valid_mask = torch.empty((len(keys), 0), dtype=torch.bool, device=device)
        return padded_idx, valid_mask

    padded_idx = torch.full((len(keys), max_len), -1, dtype=torch.long, device=device)
    valid_mask = torch.zeros((len(keys), max_len), dtype=torch.bool, device=device)

    for i, ents in enumerate(lists):
        if len(ents) > 0:
            padded_idx[i, :len(ents)] = torch.tensor(ents, dtype=torch.long, device=device)
            valid_mask[i, :len(ents)] = True

    return padded_idx, valid_mask


@torch.no_grad()
def evaluate(
    model,
    triples,
    all_triples,
    device,
    batch_size=128,
    filtered=True,
):
    model.eval()

    hr_to_t = None
    rt_to_h = None
    if filtered:
        hr_to_t, rt_to_h = build_filter_dict(all_triples)

    ranks = []

    for start in range(0, len(triples), batch_size):
        batch = triples[start:start + batch_size]
        bsz = len(batch)

        h = torch.tensor([x[0] for x in batch], dtype=torch.long, device=device)
        r = torch.tensor([x[1] for x in batch], dtype=torch.long, device=device)
        t = torch.tensor([x[2] for x in batch], dtype=torch.long, device=device)

        row_ids = torch.arange(bsz, device=device)

        # -------------------------
        # tail prediction
        # -------------------------
        tail_scores = model.score_all_tails(h, r)
        true_tail_scores = tail_scores[row_ids, t].clone()

        if filtered:
            hr_keys = [(hh, rr) for hh, rr, _ in batch]
            tail_filter_idx, tail_valid_mask = _make_padded_filter_index(hr_keys, hr_to_t, device)

            if tail_filter_idx.numel() > 0:
                # 正解 tail は消さない
                tail_valid_mask = tail_valid_mask & (tail_filter_idx != t.unsqueeze(1))

                mask_rows = row_ids.unsqueeze(1).expand_as(tail_filter_idx)[tail_valid_mask]
                mask_cols = tail_filter_idx[tail_valid_mask]
                tail_scores[mask_rows, mask_cols] = -1e9

        tail_rank = 1 + torch.sum(tail_scores > true_tail_scores.unsqueeze(1), dim=1)
        ranks.extend(tail_rank.tolist())

        # -------------------------
        # head prediction
        # -------------------------
        head_scores = model.score_all_heads(r, t)
        true_head_scores = head_scores[row_ids, h].clone()

        if filtered:
            rt_keys = [(rr, tt) for _, rr, tt in batch]
            head_filter_idx, head_valid_mask = _make_padded_filter_index(rt_keys, rt_to_h, device)

            if head_filter_idx.numel() > 0:
                # 正解 head は消さない
                head_valid_mask = head_valid_mask & (head_filter_idx != h.unsqueeze(1))

                mask_rows = row_ids.unsqueeze(1).expand_as(head_filter_idx)[head_valid_mask]
                mask_cols = head_filter_idx[head_valid_mask]
                head_scores[mask_rows, mask_cols] = -1e9

        head_rank = 1 + torch.sum(head_scores > true_head_scores.unsqueeze(1), dim=1)
        ranks.extend(head_rank.tolist())

    ranks = torch.tensor(ranks, dtype=torch.float)
    mrr = torch.mean(1.0 / ranks).item()
    hits1 = torch.mean((ranks <= 1).float()).item()
    hits3 = torch.mean((ranks <= 3).float()).item()
    hits10 = torch.mean((ranks <= 10).float()).item()

    return EvalResult(
        mrr=mrr,
        hits1=hits1,
        hits3=hits3,
        hits10=hits10,
    )

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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True, help="TSV triple file path")
    parser.add_argument("--model", type=str, default="transe", choices=["transe", "distmult"])
    parser.add_argument("--emb_dim", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--margin", type=float, default=1.0)
    parser.add_argument("--train_ratio", type=float, default=0.8)
    parser.add_argument("--valid_ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str, default="./output_kg")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    set_seed(args.seed)

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

    train_dataset = KGDataset(train_triples)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        drop_last=False,
    )

    device = torch.device(args.device)
    model = build_model(args.model, num_entities, num_relations, args.emb_dim, args.margin).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    all_known_triples = train_triples + valid_triples + test_triples
    best_valid_mrr = -1.0
    best_state = None

    print(f"=== Training {args.model} ===")
    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            device=device,
            num_entities=num_entities,
            margin=args.margin,
        )

        valid_result = evaluate(
            model=model,
            triples=valid_triples,
            all_triples=all_known_triples,
            device=device,
            batch_size=args.batch_size,
        )

        print(
            f"[Epoch {epoch:03d}] "
            f"loss={train_loss:.4f} | "
            f"valid MRR={valid_result.mrr:.4f} "
            f"H@1={valid_result.hits1:.4f} "
            f"H@3={valid_result.hits3:.4f} "
            f"H@10={valid_result.hits10:.4f}"
        )

        if valid_result.mrr > best_valid_mrr:
            best_valid_mrr = valid_result.mrr
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    print("\n=== Final Test Evaluation ===")
    test_result = evaluate(
        model=model,
        triples=test_triples,
        all_triples=all_known_triples,
        device=device,
        batch_size=args.batch_size,
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
