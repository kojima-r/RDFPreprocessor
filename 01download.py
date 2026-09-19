import os
from pathlib import Path

import requests
from tqdm import tqdm
import time

HOST = "https://rdfportal.org/ntriples"
FILE_LIST_PATH = "filelist.txt"
OUTPUT_ROOT = Path("data01")
CHUNK_SIZE = 1024
TIMEOUT=60
MAX_RETRY=5

def get_content_length(url: str) -> int | None:
    """
    URL 先のファイルサイズを取得する。
    content-length が取得できない場合は None を返す。
    """
    response = requests.head(url, allow_redirects=True, timeout=TIMEOUT)
    response.raise_for_status()

    content_length = response.headers.get("content-length")
    if content_length is None:
        return None

    return int(content_length)


def download_file(url: str, output_path: Path) -> None:
    """
    指定した URL のファイルをダウンロードして保存する。
    進捗表示には tqdm を使用する。
    """
    for _ in range(MAX_RETRY):
        try:
            file_size = get_content_length(url)
        except:
            file_size=None
            time.sleep(10)
        
    if file_size is None:
        print(f"SKIP: content-length が取得できませんでした: {url}")
        raise Exception
        return

    try:
        with requests.get(url, stream=True, timeout=TIMEOUT) as response:
            response.raise_for_status()

            with open(output_path, "wb") as file, tqdm(
                total=file_size,
                unit="B",
                unit_scale=True,
                desc=output_path.name,
            ) as progress_bar:
                for chunk in response.iter_content(chunk_size=CHUNK_SIZE):
                    # keep-alive 用の空チャンクを除外する
                    if not chunk:
                        continue

                    file.write(chunk)
                    progress_bar.update(len(chunk))
    except:
        print("[ERROR]",output_path)
        if os.path.isfile(output_path):
            os.remove(output_path)
            print(f'ファイル{output_path}を削除しました')
        else:
            print(f'{output_path}はファイルではありません')



def iter_target_files(file_list_path: str):
    """
    filelist.txt を読み込み、
    'latest' を含むパス情報を 1 行ずつ返す。
    """
    with open(file_list_path, "r", encoding="utf-8") as file:
        for line in file:
            parts = line.strip().split("/")
            if "latest" in parts:
                yield parts


def build_url_and_path(parts: list[str]) -> tuple[str, Path]:
    """
    filelist.txt の 1 行分の分割結果から、
    ダウンロード URL と保存先パスを作成する。
    """
    url = f"{HOST}/{'/'.join(parts)}"
    output_dir = OUTPUT_ROOT.joinpath(*parts[:-1])
    output_path = output_dir / parts[-1]
    return url, output_path


def main() -> None:
    """
    filelist.txt をもとに対象ファイルを順次ダウンロードする。
    """
    for parts in iter_target_files(FILE_LIST_PATH):
        url, output_path = build_url_and_path(parts)

        # 保存先ディレクトリがなければ作成する
        output_path.parent.mkdir(parents=True, exist_ok=True)

        print(f"download target: {url}")

        # 既にファイルが存在する場合は再ダウンロードしない
        if output_path.is_file():
            print(f"SKIP: 既に存在します: {output_path}")
            continue

        download_file(url, output_path)


if __name__ == "__main__":
    main()

