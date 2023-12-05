import numpy as np
import pandas as pd
import os
from multiprocessing import Pool
from tqdm import tqdm

from factor_base import FactorBase
from analysis_api import FactorAnalysis

num_jobs = 5
factor_files = ["factor_R1V2"]


analysis_config = {
    'start_date': '2016-01-01',
    'end_date': '2023-11-28',
    'factor': factor_files,
    'factor_df': None,
    'quantile': 10,
    'rebalance': 5,
    'periods': 5,
    'ret_type': 'vwap',
    'pool': 'all',
    'benchmark': '000852.SH',
    'factor_db': 'public',
    'cost': 0.0015,
    'is_adj_cost': False,
    'is_neutralize': False,
    'is_fillna': True,
    'is_winsorize': True,
    'is_normalize':True,
    'is_save': True,
    'filtered_pct': 0.15,
    'turnover_limit': 20000000,
    'result_save_dir':
    '/mnt/research/data/temp/zhangsurui/E2E_NN/Round1/Running_Results_Record/'
}

analyzer = FactorAnalysis(**analysis_config)
analyzer.multi_factor_mp(factor_files, n_process=num_jobs)

analyzer.collect_result(
    factor_files, is_remove=False, factor_batch='/mnt/research/data/temp/zhangsurui/E2E_NN/Round1/Running_Results_Record/')
