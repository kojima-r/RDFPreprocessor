import argparse
import json
import math
import os
import random
from typing import Iterator, List, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


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


def iter_tsv(path: str):
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) != 3:
                raise ValueError(
                    f"File={path}, line={line_no}: expected 3 columns, got {len(parts)}"
                )
            yield parts[0], parts[1], parts[2]


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
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    total += 1
    return total


def infer_counts_from_mapped_files(paths: List[str]) -> Tuple[int, int]:
    max_ent = -1
    max_rel = -1
    for path in paths:
        for h, r, t in iter_tsv(path):
            hi = int(h)
            ri = int(r)
            ti = int(t)
            if hi > max_ent:
                max_ent = hi
            if ti > max_ent:
                max_ent = ti
            if ri > max_rel:
                max_rel = ri
    return max_ent + 1, max_rel + 1


def batch_iterator(
    paths: List[str],
    batch_size: int,
    input_is_mapped: bool,
    ent2id: Optional[dict],
    rel2id: Optional[dict],
    shuffle_files_each_epoch: bool = True,
    seed: int = 42,
):
    files = paths[:]
    if shuffle_files_each_epoch:
        rng = random.Random(seed)
        rng.shuffle(files)

    batch_h, batch_r, batch_t = [], [], []

    for path in files:
        for h, r, t in iter_tsv(path):
            if input_is_mapped:
                hi, ri, ti = int(h), int(r), int(t)
            else:
                hi, ri, ti = ent2id[h], rel2id[r], ent2id[t]

            batch_h.append(hi)
            batch_r.append(ri)
            batch_t.append(ti)

            if len(batch_h) == batch_size:
                yield (
                    torch.tensor(batch_h, dtype=torch.long),
                    torch.tensor(batch_r, dtype=torch.long),
                    torch.tensor(batch_t, dtype=torch.long),
                )
                batch_h, batch_r, batch_t = [], [], []

    if batch_h:
        yield (
            torch.tensor(batch_h, dtype=torch.long),
            torch.tensor(batch_r, dtype=torch.long),
            torch.tensor(batch_t, dtype=torch.long),
        )


class KGEModel(nn.Module):
    def __init__(self, num_entities, num_relations, emb_dim):
        super().__init__()
        self.entity_emb = nn.Embedding(num_entities, emb_dim, sparse=True)
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
        x = h + r
        return -torch.cdist(x, self.entity_emb.weight, p=self.p_norm)

    def score_all_heads(self, r_idx, t_idx):
        r = self.relation_emb(r_idx)
        t = self.entity_emb(t_idx)
        x = t - r
        return -torch.cdist(x, self.entity_emb.weight, p=self.p_norm)


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
        return x @ self.entity_emb.weight.t()

    def score_all_heads(self, r_idx, t_idx):
        r = self.relation_emb(r_idx)
        t = self.entity_emb(t_idx)
        x = r * t
        return x @ self.entity_emb.weight.t()


def build_model(name, num_entities, num_relations, emb_dim, margin):
    name = name.lower()
    if name == "transe":
        return TransE(num_entities, num_relations, emb_dim, margin=margin, p_norm=1)
    if name == "distmult":
        return DistMult(num_entities, num_relations, emb_dim)
    raise ValueError(f"Unknown model: {name}")


def train_one_epoch(
    model,
    optimizer,
    train_paths,
    batch_size,
    input_is_mapped,
    ent2id,
    rel2id,
    num_entities,
    margin,
    device,
    epoch_seed,
):
    model.train()
    total_loss = 0.0
    total_examples = 0

    for h, r, t in batch_iterator(
        paths=train_paths,
        batch_size=batch_size,
        input_is_mapped=input_is_mapped,
        ent2id=ent2id,
        rel2id=rel2id,
        shuffle_files_each_epoch=True,
        seed=epoch_seed,
    ):
        h = h.to(device)
        r = r.to(device)
        t = t.to(device)

        optimizer.zero_grad()

        corrupt_head = torch.rand(h.size(0), device=device) < 0.5
        neg_h = h.clone()
        neg_t = t.clone()

        if corrupt_head.any():
            neg_h[corrupt_head] = torch.randint(
                0, num_entities, size=(corrupt_head.sum().item(),), device=device
            )
        if (~corrupt_head).any():
            neg_t[~corrupt_head] = torch.randint(
                0, num_entities, size=((~corrupt_head).sum().item(),), device=device
            )

        pos_score = model.score(h, r, t)
        neg_score = model.score(neg_h, r, neg_t)

        loss = F.relu(margin - pos_score + neg_score).mean()
        loss.backward()
        optimizer.step()

        if isinstance(model, TransE):
            with torch.no_grad():
                model.entity_emb.weight.data = F.normalize(model.entity_emb.weight.data, p=2, dim=1)
                model.relation_emb.weight.data = F.normalize(model.relation_emb.weight.data, p=2, dim=1)

        bs = h.size(0)
        total_loss += loss.item() * bs
        total_examples += bs

    return total_loss / max(total_examples, 1)


@torch.no_grad()
def evaluate_unfiltered(
    model,
    eval_paths,
    batch_size,
    input_is_mapped,
    ent2id,
    rel2id,
    device,
):
    model.eval()
    ranks = []

    for h, r, t in batch_iterator(
        paths=eval_paths,
        batch_size=batch_size,
        input_is_mapped=input_is_mapped,
        ent2id=ent2id,
        rel2id=rel2id,
        shuffle_files_each_epoch=False,
        seed=0,
    ):
        h = h.to(device)
        r = r.to(device)
        t = t.to(device)

        tail_scores = model.score_all_tails(h, r)
        true_tail_scores = tail_scores[torch.arange(h.size(0), device=device), t]
        tail_rank = 1 + torch.sum(tail_scores > true_tail_scores.unsqueeze(1), dim=1)
        ranks.extend(tail_rank.detach().cpu().tolist())

        head_scores = model.score_all_heads(r, t)
        true_head_scores = head_scores[torch.arange(h.size(0), device=device), h]
        head_rank = 1 + torch.sum(head_scores > true_head_scores.unsqueeze(1), dim=1)
        ranks.extend(head_rank.detach().cpu().tolist())

    ranks = torch.tensor(ranks, dtype=torch.float)
    mrr = torch.mean(1.0 / ranks).item()
    hits1 = torch.mean((ranks <= 1).float()).item()
    hits3 = torch.mean((ranks <= 3).float()).item()
    hits10 = torch.mean((ranks <= 10).float()).item()
    return {"mrr": mrr, "hits1": hits1, "hits3": hits3, "hits10": hits10}


def save_embeddings(output_dir, model, entity_mapping=None, relation_mapping=None):
    os.makedirs(output_dir, exist_ok=True)

    entity_path = os.path.join(output_dir, "entity_embeddings.tsv")
    relation_path = os.path.join(output_dir, "relation_embeddings.tsv")

    id2ent = None
    id2rel = None
    if entity_mapping is not None:
        id2ent = {v: k for k, v in entity_mapping.items()}
    if relation_mapping is not None:
        id2rel = {v: k for k, v in relation_mapping.items()}

    with open(entity_path, "w", encoding="utf-8") as f:
        f.write("raw_id\tinternal_id\tembedding\n")
        weight = model.entity_emb.weight.detach().cpu()
        for i in range(weight.size(0)):
            raw_id = id2ent[i] if id2ent is not None else str(i)
            vec = " ".join(f"{x:.8f}" for x in weight[i].tolist())
            f.write(f"{raw_id}\t{i}\t{vec}\n")

    with open(relation_path, "w", encoding="utf-8") as f:
        f.write("raw_id\tinternal_id\tembedding\n")
        weight = model.relation_emb.weight.detach().cpu()
        for i in range(weight.size(0)):
            raw_id = id2rel[i] if id2rel is not None else str(i)
            vec = " ".join(f"{x:.8f}" for x in weight[i].tolist())
            f.write(f"{raw_id}\t{i}\t{vec}\n")

    print(f"Saved entity embeddings to: {entity_path}")
    print(f"Saved relation embeddings to: {relation_path}")


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
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--margin", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output_dir", type=str, default="./outputs")
    args = parser.parse_args()

    set_seed(args.seed)

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
        if args.metadata_path is not None and os.path.exists(args.metadata_path):
            with open(args.metadata_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
            num_entities = int(meta["num_entities"])
            num_relations = int(meta["num_relations"])
            print("#entities:",num_entities)
            print("#relations:",num_relations)
        else:
            print("...infering counts")
            num_entities, num_relations = infer_counts_from_mapped_files(
                args.train_paths + args.valid_paths + args.test_paths
            )
            print("#entities:",num_entities)
            print("#relations:",num_relations)

    device = torch.device(args.device)
    model = build_model(args.model, num_entities, num_relations, args.emb_dim, args.margin).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    print("=== Dataset summary ===")
    print(f"#train triples: {count_lines(args.train_paths)}")
    print(f"#valid triples: {count_lines(args.valid_paths)}")
    print(f"#test triples : {count_lines(args.test_paths)}")
    print(f"#entities     : {num_entities}")
    print(f"#relations    : {num_relations}")

    best_valid_mrr = -1.0
    best_state = None

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(
            model=model,
            optimizer=optimizer,
            train_paths=args.train_paths,
            batch_size=args.batch_size,
            input_is_mapped=args.input_is_mapped,
            ent2id=ent2id,
            rel2id=rel2id,
            num_entities=num_entities,
            margin=args.margin,
            device=device,
            epoch_seed=args.seed + epoch,
        )

        valid_result = evaluate_unfiltered(
            model=model,
            eval_paths=args.valid_paths,
            batch_size=args.batch_size,
            input_is_mapped=args.input_is_mapped,
            ent2id=ent2id,
            rel2id=rel2id,
            device=device,
        )

        print(
            f"[Epoch {epoch:03d}] "
            f"loss={train_loss:.4f} | "
            f"valid MRR={valid_result['mrr']:.4f} "
            f"H@1={valid_result['hits1']:.4f} "
            f"H@3={valid_result['hits3']:.4f} "
            f"H@10={valid_result['hits10']:.4f}"
        )

        if valid_result["mrr"] > best_valid_mrr:
            best_valid_mrr = valid_result["mrr"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    test_result = evaluate_unfiltered(
        model=model,
        eval_paths=args.test_paths,
        batch_size=args.batch_size,
        input_is_mapped=args.input_is_mapped,
        ent2id=ent2id,
        rel2id=rel2id,
        device=device,
    )

    print("\n=== Final Test Evaluation (unfiltered) ===")
    print(f"Test MRR    : {test_result['mrr']:.4f}")
    print(f"Test Hits@1 : {test_result['hits1']:.4f}")
    print(f"Test Hits@3 : {test_result['hits3']:.4f}")
    print(f"Test Hits@10: {test_result['hits10']:.4f}")

    save_embeddings(
        output_dir=args.output_dir,
        model=model,
        entity_mapping=ent2id,
        relation_mapping=rel2id,
    )


if __name__ == "__main__":
    main()
