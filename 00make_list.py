import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin

# クロール開始URL
HOST = "https://rdfportal.org/ntriples/"


def fetch_page(url: str) -> str:
    """
    指定したURLにアクセスしてHTML文字列を取得する。

    Parameters
    ----------
    url : str
        取得対象のURL

    Returns
    -------
    str
        レスポンス本文（HTML）
    """
    print(f"Accessing to {url}...")
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    return response.text


def collect_files(url: str, path_parts: list[str], depth: int) -> list[list[str]]:
    """
    ディレクトリ一覧ページを再帰的にたどり、ファイルパスの一覧を取得する。

    Parameters
    ----------
    url : str
        現在クロール中のURL
    path_parts : list[str]
        現在の相対パスを構成する要素のリスト
    depth : int
        再帰探索の残り深さ

    Returns
    -------
    list[list[str]]
        ファイルパスを要素ごとのリストで格納した一覧
    """
    html = fetch_page(url)
    soup = BeautifulSoup(html, "html.parser")
    links = soup.find_all("a")

    results = []

    for link in links:
        href = link.get("href")

        # hrefが存在しない場合はスキップ
        if not href:
            continue

        # 親ディレクトリへのリンクは除外
        if href.startswith(".."):
            continue

        # ディレクトリの場合
        if href.endswith("/"):
            print(">>", href)

            # 深さが残っている場合のみ再帰的に探索
            if depth > 0:
                next_url = urljoin(url, href)
                child_results = collect_files(
                    next_url,
                    path_parts + [href.rstrip("/")],
                    depth - 1,
                )
                results.extend(child_results)
        else:
            # ファイルの場合はパスを保存
            print(href)
            results.append(path_parts + [href])

    return results


def save_file_list(file_paths: list[list[str]], output_path: str) -> None:
    """
    ファイル一覧をテキストファイルに保存する。

    Parameters
    ----------
    file_paths : list[list[str]]
        保存対象のファイルパス一覧
    output_path : str
        出力先ファイル名
    """
    with open(output_path, "w", encoding="utf-8") as f:
        for parts in file_paths:
            f.write("/".join(parts))
            f.write("\n")


def main() -> None:
    """
    クロールを実行し、取得したファイル一覧を保存する。
    """
    max_depth = 5
    output_file = "filelist.txt"

    all_files = collect_files(HOST, [], max_depth)
    save_file_list(all_files, output_file)


if __name__ == "__main__":
    main()
