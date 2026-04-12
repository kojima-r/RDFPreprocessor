import argparse
import json
import math
import os
import random
from dataclasses import dataclass
from typing import Dict, Iterator, List, Optional, Tuple

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

from torchrec import EmbeddingBagCollection
from torchrec.distributed import DistributedModelParallel
from torchrec.distributed.embeddingbag import EmbeddingBagCollectionSharder
from torchrec.distributed.planner import EmbeddingShardingPlanner, Topology
from torchrec.modules.embedding_configs import EmbeddingBagConfig
from torchrec.sparse.jagged_tensor import KeyedJaggedTensor


# ----------------------------
# utils
# ----------------------------

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


def init_distributed() -> Tuple[int, int, int, torch.device]:
    if "RANK" not in os.environ or "WORLD_SIZE" not in os.environ:
        raise RuntimeError(
            "This script must be launched with torchrun. "
            "Example: torchrun --nproc_per_node=4 kge_torchrec.py ..."
        )

    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ.get("LOCAL_RANK", rank))

    if torch.cuda.is_available():
        backend = "nccl"
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
    else:
        backend = "gloo"
        device = torch.device("cpu")

    dist.init_process_group(backend=backend)
    return rank, world_size, local_rank, device


def cleanup_distributed() -> None:
    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


def is_main_process() -> bool:
    return (not dist.is_initialized()) or dist.get_rank() == 0


def rank_print(*args, **kwargs):
    if is_main_process():
        print(*args, **kwargs)


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
            max_ent = max(max_ent, hi, ti)
            max_rel = max(max_rel, ri)
    return max_ent + 1, max_rel + 1


def shard_paths(paths: List[str], rank: int, world_size: int) -> List[str]:
    # ファイル単位で雑に shard。ファイル数が少ない場合は後述の iterator 内で行単位 shard も可。
    return [p for i, p in enumerate(paths) if i % world_size == rank]


# ----------------------------
# dataloader
# ----------------------------

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


# ----------------------------
# TorchRec feature builder
# ----------------------------

ENTITY_FEATURE_KEYS = ["head", "tail", "neg_head", "neg_tail"]


def single_id_kjt(
    features: Dict[str, torch.Tensor],
    device: torch.device,
) -> KeyedJaggedTensor:
    """
    各 feature は shape=[B] の entity id。
    bag size = 1 として EmbeddingBagCollection に流す。
    """
    keys = []
    values = []
    lengths = []

    for key in ENTITY_FEATURE_KEYS:
        if key in features:
            x = features[key].reshape(-1).to(device=device, dtype=torch.long)
            keys.append(key)
            values.append(x)
            lengths.append(torch.ones_like(x, dtype=torch.int32, device=device))

    return KeyedJaggedTensor.from_lengths_sync(
        keys=keys,
        values=torch.cat(values, dim=0),
        lengths=torch.cat(lengths, dim=0),
    )


# ----------------------------
# model
# ----------------------------
from torchrec import PoolingType
class ShardedEntityEncoder(nn.Module):
    """
    1つの entity table を 4 feature(head/tail/neg_head/neg_tail) から参照する。
    weight 自体は1つなので、head/tail で重み共有される。
    """
    
    def __init__(self, num_entities: int, emb_dim: int):
        super().__init__()
        self.ebc = EmbeddingBagCollection(
            tables=[
                EmbeddingBagConfig(
                    name="entity_table",
                    embedding_dim=emb_dim,
                    num_embeddings=num_entities,
                    feature_names=ENTITY_FEATURE_KEYS,
                    #pooling="sum",  # bag size=1 なので実質 lookup
                    pooling=PoolingType.SUM,
                )
            ],
            is_weighted=False,
            device=torch.device("meta"),  # planner/DMP 用
        )

    def forward(self, entity_kjt: KeyedJaggedTensor) -> Dict[str, torch.Tensor]:
        pooled = self.ebc(entity_kjt)
        # pooled は KeyedTensor。各 feature の embedding を取り出す
        out = {}
        for key in ENTITY_FEATURE_KEYS:
            if key in pooled.keys():
                out[key] = pooled[key]
        return out


class KGETorchRecModel(nn.Module):
    def __init__(
        self,
        num_entities: int,
        num_relations: int,
        emb_dim: int,
        model_name: str,
        margin: float = 1.0,
        p_norm: int = 1,
        relation_sparse: bool = False,
    ):
        super().__init__()
        self.model_name = model_name.lower()
        self.margin = margin
        self.p_norm = p_norm
        self.emb_dim = emb_dim

        self.entity_encoder = ShardedEntityEncoder(
            num_entities=num_entities,
            emb_dim=emb_dim,
        )
        # relation は通常そこまで巨大でないので複製保持
        self.relation_emb = nn.Embedding(num_relations, emb_dim, sparse=relation_sparse)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.relation_emb.weight)

    def score_triples(
        self,
        head_e: torch.Tensor,
        rel_e: torch.Tensor,
        tail_e: torch.Tensor,
    ) -> torch.Tensor:
        if self.model_name == "transe":
            return -torch.norm(head_e + rel_e - tail_e, p=self.p_norm, dim=-1)
        elif self.model_name == "distmult":
            return torch.sum(head_e * rel_e * tail_e, dim=-1)
        else:
            raise ValueError(f"Unknown model: {self.model_name}")

    def forward(
        self,
        h: torch.Tensor,
        r: torch.Tensor,
        t: torch.Tensor,
        neg_h: torch.Tensor,
        neg_t: torch.Tensor,
        device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        kjt = single_id_kjt(
            {
                "head": h,
                "tail": t,
                "neg_head": neg_h,
                "neg_tail": neg_t,
            },
            device=device,
        )
        entity_out = self.entity_encoder(kjt)

        head_e = entity_out["head"]
        tail_e = entity_out["tail"]
        neg_head_e = entity_out["neg_head"]
        neg_tail_e = entity_out["neg_tail"]

        rel_e = self.relation_emb(r)

        pos_score = self.score_triples(head_e, rel_e, tail_e)
        neg_score = self.score_triples(neg_head_e, rel_e, neg_tail_e)
        return pos_score, neg_score


# ----------------------------
# distributed model build
# ----------------------------
def build_sharded_model(
    num_entities: int,
    num_relations: int,
    emb_dim: int,
    model_name: str,
    margin: float,
    device: torch.device,
    world_size: int,
):
    if world_size <= 1:
        raise RuntimeError(
            "TorchRec sharding requires world_size > 1 for this model size."
        )

    # ここでは .to(device) しない
    base_model = KGETorchRecModel(
        num_entities=num_entities,
        num_relations=num_relations,
        emb_dim=emb_dim,
        model_name=model_name,
        margin=margin,
    )

    # meta ではない dense 側だけ GPU へ
    base_model.relation_emb = base_model.relation_emb.to(device)

    sharder = EmbeddingBagCollectionSharder()

    planner = EmbeddingShardingPlanner(
        topology=Topology(
            world_size=world_size,
            compute_device="cuda" if device.type == "cuda" else "cpu",
        ),
    )

    plan = planner.collective_plan(
        base_model.entity_encoder,
        [sharder],
        dist.group.WORLD,
    )

    sharded_entity_encoder = DistributedModelParallel(
        module=base_model.entity_encoder,
        device=device,
        plan=plan,
        sharders=[sharder],
        init_data_parallel=False,
    )

    base_model.entity_encoder = sharded_entity_encoder
    return base_model

# ----------------------------
# train / eval
# ----------------------------

def sample_negative_entities(
    h: torch.Tensor,
    t: torch.Tensor,
    num_entities: int,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    corrupt_head = torch.rand(h.size(0), device=device) < 0.5
    neg_h = h.clone()
    neg_t = t.clone()

    if corrupt_head.any():
        neg_h[corrupt_head] = torch.randint(
            0,
            num_entities,
            size=(int(corrupt_head.sum().item()),),
            device=device,
        )
    if (~corrupt_head).any():
        neg_t[~corrupt_head] = torch.randint(
            0,
            num_entities,
            size=(int((~corrupt_head).sum().item()),),
            device=device,
        )
    return neg_h, neg_t


def train_one_epoch(
    model: nn.Module,
    optimizer,
    train_paths: List[str],
    batch_size: int,
    input_is_mapped: bool,
    ent2id: Optional[dict],
    rel2id: Optional[dict],
    num_entities: int,
    margin: float,
    device: torch.device,
    epoch_seed: int,
    rank: int,
    world_size: int,
):
    model.train()
    total_loss = 0.0
    total_examples = 0

    local_paths = shard_paths(train_paths, rank, world_size)

    for h, r, t in batch_iterator(
        paths=local_paths,
        batch_size=batch_size,
        input_is_mapped=input_is_mapped,
        ent2id=ent2id,
        rel2id=rel2id,
        shuffle_files_each_epoch=True,
        seed=epoch_seed + rank,
    ):
        h = h.to(device, non_blocking=True)
        r = r.to(device, non_blocking=True)
        t = t.to(device, non_blocking=True)

        neg_h, neg_t = sample_negative_entities(h, t, num_entities, device)

        optimizer.zero_grad(set_to_none=True)
        pos_score, neg_score = model(h, r, t, neg_h, neg_t, device)

        loss = F.relu(margin - pos_score + neg_score).mean()
        loss.backward()
        
        #手動 all-reduce
        for p in model.relation_emb.parameters():
            if p.grad is not None:
                dist.all_reduce(p.grad, op=dist.ReduceOp.SUM)
                p.grad /= dist.get_world_size()

        optimizer.step()

        bs = h.size(0)
        total_loss += loss.item() * bs
        total_examples += bs

    # rank 間集約
    stats = torch.tensor([total_loss, total_examples], device=device, dtype=torch.float64)
    dist.all_reduce(stats, op=dist.ReduceOp.SUM)
    total_loss = float(stats[0].item())
    total_examples = int(stats[1].item())
    return total_loss / max(total_examples, 1)


@torch.no_grad()
def evaluate_sampled(
    model: nn.Module,
    eval_paths: List[str],
    batch_size: int,
    input_is_mapped: bool,
    ent2id: Optional[dict],
    rel2id: Optional[dict],
    num_entities: int,
    device: torch.device,
    rank: int,
    world_size: int,
    num_negatives: int = 128,
):
    """
    巨大グラフ向けに、全 entity 総当たりの代わりに sampled ranking を返す。
    MRR/Hits は元の unfiltered exact ranking とは一致しない点に注意。
    """
    model.eval()
    ranks = []

    local_paths = shard_paths(eval_paths, rank, world_size)

    for h, r, t in batch_iterator(
        paths=local_paths,
        batch_size=batch_size,
        input_is_mapped=input_is_mapped,
        ent2id=ent2id,
        rel2id=rel2id,
        shuffle_files_each_epoch=False,
        seed=0,
    ):
        h = h.to(device, non_blocking=True)
        r = r.to(device, non_blocking=True)
        t = t.to(device, non_blocking=True)

        bsz = h.size(0)

        # true triple score
        pos_h = h
        pos_t = t
        pos_score, _ = model(
            pos_h, r, pos_t,
            neg_h=pos_h,  # dummy
            neg_t=pos_t,  # dummy
            device=device,
        )
        true_score = pos_score

        # tail corruption sampled ranking
        sampled_neg_t = torch.randint(
            low=0,
            high=num_entities,
            size=(bsz, num_negatives),
            device=device,
        )

        greater_count = torch.zeros(bsz, device=device, dtype=torch.long)
        for j in range(num_negatives):
            cur_neg_t = sampled_neg_t[:, j]
            _, neg_score = model(
                h, r, t,
                neg_h=h,          # dummy
                neg_t=cur_neg_t,
                device=device,
            )
            greater_count += (neg_score > true_score).long()

        tail_rank = 1 + greater_count
        ranks.append(tail_rank)

    if len(ranks) == 0:
        local_ranks = torch.empty(0, device=device, dtype=torch.float32)
    else:
        local_ranks = torch.cat(ranks).float()

    # 可変長 gather
    local_n = torch.tensor([local_ranks.numel()], device=device, dtype=torch.long)
    all_n = [torch.zeros_like(local_n) for _ in range(world_size)]
    dist.all_gather(all_n, local_n)
    max_n = int(max(x.item() for x in all_n))

    padded = torch.full((max_n,), -1.0, device=device)
    if local_ranks.numel() > 0:
        padded[: local_ranks.numel()] = local_ranks

    gathered = [torch.empty_like(padded) for _ in range(world_size)]
    dist.all_gather(gathered, padded)

    if is_main_process():
        flat = torch.cat(gathered)
        flat = flat[flat > 0]
        mrr = torch.mean(1.0 / flat).item() if flat.numel() > 0 else 0.0
        hits1 = torch.mean((flat <= 1).float()).item() if flat.numel() > 0 else 0.0
        hits3 = torch.mean((flat <= 3).float()).item() if flat.numel() > 0 else 0.0
        hits10 = torch.mean((flat <= 10).float()).item() if flat.numel() > 0 else 0.0
        return {"mrr": mrr, "hits1": hits1, "hits3": hits3, "hits10": hits10}
    else:
        return None


# ----------------------------
# checkpoint / export
# ----------------------------

def save_relation_embeddings(output_dir: str, model: nn.Module, relation_mapping=None):
    if not is_main_process():
        return

    os.makedirs(output_dir, exist_ok=True)
    relation_path = os.path.join(output_dir, "relation_embeddings.tsv")

    id2rel = None
    if relation_mapping is not None:
        id2rel = {v: k for k, v in relation_mapping.items()}

    weight = model.relation_emb.weight.detach().cpu()

    with open(relation_path, "w", encoding="utf-8") as f:
        f.write("raw_id\tinternal_id\tembedding\n")
        for i in range(weight.size(0)):
            raw_id = id2rel[i] if id2rel is not None else str(i)
            vec = " ".join(f"{x:.8f}" for x in weight[i].tolist())
            f.write(f"{raw_id}\t{i}\t{vec}\n")

    print(f"Saved relation embeddings to: {relation_path}")


def save_local_entity_shard(output_dir: str, rank: int, model: nn.Module):
    """
    TorchRec shard は rank ごとにローカル保存する。
    巨大 entity table を全 gather しない。
    """
    os.makedirs(output_dir, exist_ok=True)
    shard_path = os.path.join(output_dir, f"entity_shard_rank{rank:04d}.pt")

    state = model.entity_encoder.state_dict()
    torch.save(state, shard_path)

    if is_main_process():
        print(f"Saved sharded entity checkpoints under: {output_dir}")


# ----------------------------
# main
# ----------------------------

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
    parser.add_argument("--emb_dim", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=4096)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--margin", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--output_dir", type=str, default="./outputs")
    parser.add_argument("--num_eval_negatives", type=int, default=128)

    args = parser.parse_args()

    rank, world_size, local_rank, device = init_distributed()
    set_seed(args.seed + rank)

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
        else:
            if rank == 0:
                print("...inferring counts from mapped files")
            num_entities, num_relations = infer_counts_from_mapped_files(
                args.train_paths + args.valid_paths + args.test_paths
            )

    if rank == 0:
        print("=== Dataset summary ===")
        #print(f"#train triples: {count_lines(args.train_paths)}")
        #print(f"#valid triples: {count_lines(args.valid_paths)}")
        #print(f"#test triples : {count_lines(args.test_paths)}")
        print(f"#entities     : {num_entities}")
        print(f"#relations    : {num_relations}")
        print(f"#world_size   : {world_size}")
        print(f"device        : {device}")

    model = build_sharded_model(
        num_entities=num_entities,
        num_relations=num_relations,
        emb_dim=args.emb_dim,
        model_name=args.model,
        margin=args.margin,
        device=device,
        world_size=world_size,
    )

    optimizer = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
    )

    best_valid_mrr = -1.0
    best_ckpt_path = os.path.join(args.output_dir, "best_dense_state_rank0.pt")

    os.makedirs(args.output_dir, exist_ok=True)

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
            rank=rank,
            world_size=world_size,
        )

        valid_result = evaluate_sampled(
            model=model,
            eval_paths=args.valid_paths,
            batch_size=args.batch_size,
            input_is_mapped=args.input_is_mapped,
            ent2id=ent2id,
            rel2id=rel2id,
            num_entities=num_entities,
            device=device,
            rank=rank,
            world_size=world_size,
            num_negatives=args.num_eval_negatives,
        )

        if is_main_process():
            print(
                f"[Epoch {epoch:03d}] "
                f"loss={train_loss:.4f} | "
                f"valid(sampled) MRR={valid_result['mrr']:.4f} "
                f"H@1={valid_result['hits1']:.4f} "
                f"H@3={valid_result['hits3']:.4f} "
                f"H@10={valid_result['hits10']:.4f}"
            )

            if valid_result["mrr"] > best_valid_mrr:
                best_valid_mrr = valid_result["mrr"]
                torch.save(
                    {
                        "relation_emb": model.relation_emb.state_dict(),
                        "best_valid_mrr": best_valid_mrr,
                        "epoch": epoch,
                    },
                    best_ckpt_path,
                )

        dist.barrier()

    test_result = evaluate_sampled(
        model=model,
        eval_paths=args.test_paths,
        batch_size=args.batch_size,
        input_is_mapped=args.input_is_mapped,
        ent2id=ent2id,
        rel2id=rel2id,
        num_entities=num_entities,
        device=device,
        rank=rank,
        world_size=world_size,
        num_negatives=args.num_eval_negatives,
    )

    if is_main_process():
        print("\n=== Final Test Evaluation (sampled ranking) ===")
        print(f"Test MRR    : {test_result['mrr']:.4f}")
        print(f"Test Hits@1 : {test_result['hits1']:.4f}")
        print(f"Test Hits@3 : {test_result['hits3']:.4f}")
        print(f"Test Hits@10: {test_result['hits10']:.4f}")

    save_relation_embeddings(
        output_dir=args.output_dir,
        model=model,
        relation_mapping=rel2id,
    )
    save_local_entity_shard(
        output_dir=args.output_dir,
        rank=rank,
        model=model,
    )

    cleanup_distributed()


if __name__ == "__main__":
    main()
