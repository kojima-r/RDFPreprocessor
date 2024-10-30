import json
import time

import requests
from bs4 import BeautifulSoup

host="https://rdfportal.org/download/"


def crawl_list_page(start_url):
    """
    記事一覧ページをクロールして記事詳細の URL を全て取得する
    :return:
    """
    print(f"Accessing to {start_url}...")
    # https://note.lapras.com/ へアクセスする
    response = requests.get(start_url)
    response.raise_for_status()
    #time.sleep(10)
    return response.text

def get_all_file(url, base, depth):
    text=crawl_list_page(url)
    parse_html = BeautifulSoup(text, "html.parser")
    title_lists = parse_html.find_all("a")
    data=[]
    for el in title_lists:
        if el.attrs["href"][:2]!="..":
            if el.attrs["href"][-1]=="/":
                #data.append((el.string, el.attrs["href"]))
                print(">>",el.attrs["href"])
                if depth>0:
                    pre_data=get_all_file(url+"/"+el.attrs["href"],base+[el.attrs["href"]], depth-1)
                    data.extend(pre_data)
            else:
                data.append(base+[el.attrs["href"]])
                print(el.attrs["href"])
    return data
out_data=get_all_file(host, [], 2)
with open("filelist.txt","w") as ofp:
    for lnk in out_data:
        ofp.write("/".join(lnk))
        ofp.write("\n")
