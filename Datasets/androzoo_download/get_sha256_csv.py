#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
    @Time    :   2023/4/12,
    @Author  :   Dirtyboy,
    @Version :   1.0,
'''

import os
import sys
import json
import pandas as pd
import argparse
import logging
import time
from datetime import datetime
from tqdm import tqdm
import asyncio
#import aiohttp
#from tenacity import retry, stop_after_attempt, before_log

parser = argparse.ArgumentParser(description='Androzoo downloader script.')
parser.add_argument('--year', type=int, help='Choose a specific year.')
parser.add_argument('--update', type=bool, default=False, help='Ignore the downloaded.')
parser.add_argument('--max', type=int, default=104000, help='Max number of apks to download.')
parser.add_argument('--coroutine', type=int, default=20, help='Number of coroutines.')
parser.add_argument('--markets', nargs='+', default=['play.google.com', 'anzhi', 'appchina'], help='Number of coroutines.')
parser.add_argument('--vt_detection', type=int, default=0, help='Download Benign apks by default. Lower bound (included) of `Malware` if greater than 0.')
parser.add_argument('--upper', type=int, help='Upper bound (not included) for `Malware`. Useful only if vt_detection is greater than 0.')
parser.add_argument('--output', type=str, default='data1', help='Save apks in /<output>/Androzoo/<Benign or Malware_LB>/<year>.')
parser.add_argument('--debug', type=bool, default=False, help='Logging level: INFO by default, DEBUG will log for every apk in both stdout and file.')
parser.add_argument('--fix', type=bool, default=False, help='Just remove the broken apks since last stop.')
parser.add_argument('--config', type=str, default='config', help='Sepecify the name for config file.')

args = parser.parse_args()
year = args.year

if args.vt_detection == 0:
    cat = 'Benign'
    print('[AndrozooDownloader] Benign Samples.')
else:
    cat = ('Malware_%d-%d' % (args.vt_detection, args.upper)) if args.upper else ('Malware_%d' % args.vt_detection)
    if args.upper:
        print('[AndrozooDownloader] Malware Samples (%d<=vt_detection<%d).' % (args.vt_detection, args.upper))
    else:
        print('[AndrozooDownloader] Malware Samples (vt_detection>=%d).' % args.vt_detection)
outdir = '%s/Androzoo/%s/%d' % (args.output, cat, year)
if not os.path.exists(outdir):
    os.makedirs(outdir)

timestamp = int(round(time.time() * 1000))
tag = '%d_%s_%d' % (year, cat, timestamp)
log_file = '%s.log' % tag
LOG_FORMAT = '%(asctime)s - %(levelname)s - %(message)s'
fmt = logging.Formatter(LOG_FORMAT)

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
if not args.debug:
    filelog_level = logging.INFO
else:
    filelog_level = logging.DEBUG
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setLevel(logging.DEBUG)
    stream_handler.setFormatter(fmt)
    logger.addHandler(stream_handler)
file_handler = logging.FileHandler(log_file)
file_handler.setLevel(filelog_level)
file_handler.setFormatter(fmt)
logger.addHandler(file_handler)


def read_config(fname='config'):
    with open(fname, 'r') as f:
        config = json.load(f)
    return config


def filter(year, a, processed):
    if cat == 'Benign':
        a = a[a.vt_detection == 0]
    else:
        a = a[a.vt_detection >= args.vt_detection]
        if args.upper:
            a = a[a.vt_detection < args.upper]
    date = pd.to_datetime(a['dex_date'])
    a = a.assign(dex_date=date)
    a = a[(a.dex_date > datetime(year,1,1)) & (a.dex_date < datetime(year+1,1,1))]

    print('[AndrozooDownloader] Selecting from markets: ', args.markets)
    pattern = ' | '.join(["(a.markets.str.contains('%s'))" % i for i in args.markets])
    # (a.markets.str.contains('play.google.com')) | (a.markets.str.contains('anzhi')) | (a.markets.str.contains('anzhi'))
    a = a[eval(pattern)]
    logging.info('%d APKs in total for year %d' % (len(a), year))
    a.to_csv(processed, index=False)
    return a


if __name__ == '__main__':
    config = read_config(args.config)

    processed = '%d_%s.csv' % (year, cat)
    if not os.path.exists(processed):
        meta = pd.read_csv(config['meta'])
        meta = filter(year, meta, processed)
    else:
        meta = pd.read_csv(processed)
    
