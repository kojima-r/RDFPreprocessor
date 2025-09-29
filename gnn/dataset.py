
import os
import torch
from torch_geometric.data import Data, InMemoryDataset

def load_edge_list_tsv(
    tsv_path: str,
    has_header: bool = False,
    undirected: bool = False,
    assume_zero_based: bool = None,
    remap_non_contiguous: bool = True,
):
    """
    3列TSV: src, edge_id, dst を読み込み、edge_index と edge_id テンソルを返します。
    - has_header: 先頭行がヘッダの場合 True
    - undirected: 無向グラフとして双方向エッジを追加
    - assume_zero_based:
        - True: 入力ノードIDは 0 始まりとみなす
        - False: 入力ノードIDは 1 始まりとみなす（すべて -1 シフト）
        - None: 自動判定（最小IDが0なら0始まり、1なら1始まりとして処理）
    - remap_non_contiguous:
        - True: ノードIDが飛び番/任意の整数でも 0..N-1 に詰め替える
    """
    src_list, eid_list, dst_list = [], [], []

    with open(tsv_path, "r", encoding="utf-8") as f:
        if has_header:
            next(f)
        for line in f:
            line = line.strip()
            if not line:
                continue
            cols = line.split("\t")
            if len(cols) < 3:
                raise ValueError(f"3列必要ですが: {line}")
            s, e, d = cols[0], cols[1], cols[2]
            src_list.append(int(s))
            eid_list.append(int(e))
            dst_list.append(int(d))

    src = torch.tensor(src_list, dtype=torch.long)
    dst = torch.tensor(dst_list, dtype=torch.long)
    edge_id = torch.tensor(eid_list, dtype=torch.long)

    # 0/1始まりの自動判定
    if assume_zero_based is None:
        min_id = min(int(src.min().item()), int(dst.min().item()))
        assume_zero_based = (min_id == 0)

    if not assume_zero_based:
        # 1始まりとみなし、0始まりに変換
        src = src - 1
        dst = dst - 1

    if remap_non_contiguous:
        # ノードIDが非連続でも 0..N-1 に詰め替え
        uniq = torch.unique(torch.cat([src, dst], dim=0))
        # 古いID -> 新しいID
        new_id = {int(old): i for i, old in enumerate(uniq.tolist())}
        src = torch.tensor([new_id[int(x)] for x in src.tolist()], dtype=torch.long)
        dst = torch.tensor([new_id[int(x)] for x in dst.tolist()], dtype=torch.long)

    # 無向＝逆向きエッジを付与（edge_id も複製）
    if undirected:
        src = torch.cat([src, dst], dim=0)
        dst = torch.cat([dst, src[:len(dst)]], dim=0)  # 注意: ここで src を上書きしているので順序に配慮
        edge_id = torch.cat([edge_id, edge_id], dim=0)

    edge_index = torch.stack([src, dst], dim=0)  # [2, E]
    num_nodes = int(torch.max(edge_index).item()) + 1 if edge_index.numel() > 0 else 0

    return edge_index, edge_id, num_nodes


def build_data_from_tsv(
    tsv_path: str,
    has_header: bool = False,
    undirected: bool = False,
    assume_zero_based: bool = None,
    remap_non_contiguous: bool = True,
) -> Data:
    edge_index, edge_id, num_nodes = load_edge_list_tsv(
        tsv_path,
        has_header=has_header,
        undirected=undirected,
        assume_zero_based=assume_zero_based,
        remap_non_contiguous=remap_non_contiguous,
    )

    data = Data(
        edge_index=edge_index,       # [2, E]
        edge_id=edge_id,             # [E]  エッジIDを属性として保持（必要に応じて edge_attr にも可）
        num_nodes=num_nodes,
    )
    return data


class SingleGraphTSVDataset(InMemoryDataset):
    """
    単一のTSVから 1つのグラフ Data を作って保持する簡易 InMemoryDataset。
    PyG の標準フロー（raw/processed）に乗せたい場合は raw_dir/processed_dir を使ってもOKですが、
    ここでは最小限でメモリ常駐にしています。
    """
    def __init__(
        self,
        tsv_path: str,
        has_header: bool = False,
        undirected: bool = False,
        assume_zero_based: bool = None,
        remap_non_contiguous: bool = True,
        transform=None,
        pre_transform=None,
    ):
        self.tsv_path = tsv_path
        self.has_header = has_header
        self.undirected = undirected
        self.assume_zero_based = assume_zero_based
        self.remap_non_contiguous = remap_non_contiguous
        super().__init__(".", transform, pre_transform)

        data = build_data_from_tsv(
            tsv_path=self.tsv_path,
            has_header=self.has_header,
            undirected=self.undirected,
            assume_zero_based=self.assume_zero_based,
            remap_non_contiguous=self.remap_non_contiguous,
        )

        if self.pre_transform is not None:
            data = self.pre_transform(data)

        self.data, self.slices = self.collate([data])

    def _download(self):
        # 何もしない
        pass

    def _process(self):
        # 何もしない（すでに __init__ で構築）
        pass

    @property
    def raw_file_names(self):
        return [os.path.basename(self.tsv_path)]

    @property
    def processed_file_names(self):
        # 実際にはファイルを作らないが、規定メソッドとして最低限定義
        return ["in_memory.pt"]


# ===== 使い方 =====
if __name__ == "__main__":
    # 例: tsv の各行が "src \t edge_id \t dst"
    # 1) 直接 Data を作成
    filename="../data06/pubchem.graph.tsv"
    data = build_data_from_tsv(filename, has_header=False, undirected=False)
    print(data)
    print(data.edge_index.shape, data.num_nodes, data.edge_id[:5] if data.edge_id.numel() else None)

    # 2) Dataset として扱う
    dataset = SingleGraphTSVDataset(filename, has_header=False, undirected=False)
    print(len(dataset), dataset[0])

