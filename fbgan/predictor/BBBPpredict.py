# -- coding: utf-8 --
# author : TangQiang
# time   : 2023/8/8
# email  : tangqiang.0701@gmail.com
# file   : BBBPpredict.py

import requests
from lxml import etree
import pandas as pd
from utils.utils import *

url = "http://i.uestc.edu.cn/BBPpredict/cgi-bin/BBPpredict.pl"
header_list = [
    {"user-agent": "Mozilla/4.0 (compatible; MSIE 7.0; Windows NT 5.1; Maxthon 2.0)"},# 遨游
    {"user-agent": "Mozilla/5.0 (Windows NT 6.1; rv:2.0.1) Gecko/20100101 Firefox/4.0.1"},# 火狐
    {"user-agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_7_0) AppleWebKit/535.11 (KHTML, like Gecko) Chrome/17.0.963.56 Safari/535.11"}# 谷歌
]

def get_BBBPpredict(seqs, df, logger, predictor_path):
    logger.info("begin run BBBPpredict")
    payload = {'pep': seqs, 'threhold': '0.5'}
    files = []
    header = random.choice(header_list)
    response = requests.request("POST", url, headers=header, data=payload, files=files)
    parser = etree.HTMLParser(encoding='utf-8')
    html = etree.HTML(response.text, parser=parser)
    table = html.xpath('//table[@id="tablesort"]')
    table = etree.tostring(table[0], encoding='utf-8').decode()
    df1 = pd.read_html(table, encoding='utf-8', header=0)[0]
    df1 = df1[['Query Sequence', 'Probability']]
    df1.rename(columns={'Query Sequence': 'peptides', 'Probability': 'prob_BBBPpredict'}, inplace=True)
    df['BBBPpredict'] = df1
    logger.info(f'end for BBBPpredict: \n {df1}')
    return None