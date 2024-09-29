# coding:utf-8
from tqdm import tqdm
import requests
import sys
import os


def txt_to_list(txt_path):
    f = open(txt_path, "r")
    return f.read().splitlines()


url = 'https://androzoo.uni.lu/api/download?apikey=&sha256={}'


def download(year):
    # path = '{}_benign.txt'.format(year)
    path = 'tmp.txt'

    a = txt_to_list(path)[6200:6280]

    for i, item in tqdm(enumerate(a), total=len(a), desc='Download samples'):
        apath = 'androzoo/{}/benign/{}.apk'.format(year, item)
        if os.path.exists(apath):
            continue

        downurl = url.format(item)
        while True:
            try:
                r = requests.get(url=downurl, timeout=30)
                break
            except Exception:
                continue

        with open(apath, 'wb') as f:
            f.write(r.content)
    return


if __name__ == '__main__':
    year = sys.argv[1:][0]
    print(year)
    download(year)
