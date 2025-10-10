import math
import torch
from torch_geometric.data import Data
from torch_geometric.nn.models import Node2Vec
from torch_geometric.utils import negative_sampling
import torch_geometric.transforms as T

from dataset import SingleGraphTSVDataset

@torch.no_grad()
def evaluate_auc(z, pos_edge_index, neg_edge_index):
    # 内積スコア
    pos_scores = (z[pos_edge_index[0]] * z[pos_edge_index[1]]).sum(dim=1)
    neg_scores = (z[neg_edge_index[0]] * z[neg_edge_index[1]]).sum(dim=1)

    scores = torch.cat([pos_scores, neg_scores], dim=0)
    order = torch.argsort(scores)
    ranks = torch.empty_like(order, dtype=torch.float)
    ranks[order] = torch.arange(1, order.numel() + 1, dtype=torch.float, device=scores.device)
    n_pos, n_neg = pos_scores.size(0), neg_scores.size(0)
    sum_ranks_pos = ranks[:n_pos].sum()
    auc = (sum_ranks_pos - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg + 1e-8)
    return float(auc)

@torch.no_grad()
def evaluate_ranking(z, pos_edge_index, num_nodes, k=10, neg_per_pos=100, seed=1234):
    g = torch.Generator().manual_seed(seed)
    ranks, hits, ndcgs = [], [], []

    for u, v in pos_edge_index.t().tolist():
        pos_score = float((z[u] * z[v]).sum().item())
        # ランダム負例サンプリング
        neg_dst = torch.randint(num_nodes, (neg_per_pos,), generator=g)
        neg_scores = (z[u] * z[neg_dst]).sum(dim=1)

        rank = int((neg_scores > pos_score).sum().item()) + 1
        ranks.append(rank)
        hits.append(1 if rank <= k else 0)
        ndcgs.append(1 / math.log2(1 + rank) if rank <= k else 0.0)

    mrr = sum(1/r for r in ranks) / len(ranks)
    hitk = sum(hits) / len(hits)
    ndcgk = sum(ndcgs) / len(ndcgs)
    return float(mrr), float(hitk), float(ndcgk)

# --- 追加: ラベルからpos/negに分けるユーティリティ ---
def split_pos_neg_from_edge_labels(split_data):
    # split_data: val_data or test_data
    idx = split_data.edge_label_index      # [2, M]
    y   = split_data.edge_label            # [M], 1=pos, 0=neg
    pos = idx[:, y == 1]
    neg = idx[:, y == 0]
    return pos, neg

@torch.no_grad()
def evaluate_auc_from_split(z, split_data):
    pos_idx, neg_idx = split_pos_neg_from_edge_labels(split_data)
    return evaluate_auc(z, pos_idx, neg_idx)

@torch.no_grad()
def evaluate_ranking_from_split(z, split_data, num_nodes, k=10, neg_per_pos=100):
    pos_idx, _ = split_pos_neg_from_edge_labels(split_data)
    return evaluate_ranking(z, pos_idx, num_nodes, k=k, neg_per_pos=neg_per_pos)

def main():
    # ===== データ読み込み =====
    filenames=[
            #"../data06/pubchem.graph.tsv",
            "../data06/pubmed.graph.tsv"]
    dataset = SingleGraphTSVDataset(filenames, has_header=False, undirected=False)
    data: Data = dataset[0]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ===== train/test split =====
    transform = T.RandomLinkSplit(
        num_val=0.05,
        num_test=0.1,
        is_undirected=True,   # グラフが無向なら True
        add_negative_train_samples=False  # Node2Vec の学習では自分で負例を作るので False
    )
    train_data, val_data, test_data = transform(data)

    # Node2Vec モデル
    model = Node2Vec(
        edge_index=train_data.edge_index,  # 学習用エッジ
        embedding_dim=128,
        walk_length=20,
        context_size=10,
        walks_per_node=10,
        num_negative_samples=1,
        sparse=True,
    ).to(device)

    loader = model.loader(batch_size=128, shuffle=True, num_workers=0)
    optimizer = torch.optim.SparseAdam(list(model.parameters()), lr=0.01)

    # 学習ループ
    n_epoch=100
    for epoch in range(1, n_epoch):
        model.train()
        total_loss = 0
        for pos_rw, neg_rw in loader:
            optimizer.zero_grad()
            loss = model.loss(pos_rw.to(device), neg_rw.to(device))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        # 評価
        model.eval()
        z = model().detach()
        
        #auc = evaluate_auc(z, test_data.pos_edge_label_index, test_data.neg_edge_label_index)
        #mrr, hit10, ndcg10 = evaluate_ranking(z, test_data.pos_edge_label_index, data.num_nodes, k=10)
        auc = evaluate_auc_from_split(z, test_data)
        mrr, hit10, ndcg10 = evaluate_ranking_from_split(
            z, test_data, data.num_nodes, k=10, neg_per_pos=100
            )


        print(f"Epoch {epoch:03d} | loss={total_loss:.4f} "
              f"| AUC={auc:.4f} | MRR={mrr:.4f} | Hit@10={hit10:.4f} | nDCG@10={ndcg10:.4f}")

if __name__ == "__main__":
    main()

