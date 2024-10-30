import os
import requests
from tqdm import tqdm

host="https://rdfportal.org/download/"

def download(url, filename):
    if "content-length" not in requests.head(url).headers:
        print("SKIP:", requests.head(url).headers)
        return
    file_size = int(requests.head(url).headers["content-length"])
    res = requests.get(url, stream=True)
    pbar = tqdm(total=file_size, unit="B", unit_scale=True)
    with open(filename, 'wb') as file:
        for chunk in res.iter_content(chunk_size=1024):
            file.write(chunk)
            pbar.update(len(chunk))
        pbar.close()
    #urlData = requests.get(url).content
    #with open(filename ,mode='wb') as fp: # wb でバイト型を書き込める
    #  fp.write(urlData)


db_list=[]
for line in open("filelist.txt"):
    arr=line.strip().split("//")
    if "latest" in arr:
        url=host+"/".join(arr)
        d="data01/"+"/".join(arr[:-1])
        os.makedirs(d,exist_ok=True)
        print("...",url)
        filename= d+"/"+arr[-1]
        if not os.path.isfile(filename):
            download(url, filename)
              

