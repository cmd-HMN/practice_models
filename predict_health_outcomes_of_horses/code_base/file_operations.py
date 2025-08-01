import pickle
import pandas as pd
from configs import cfg

def read_data(file, cfg=cfg):
    try:
        return pd.read_csv(f'{cfg.BASE_PATH}/{file}.csv')
    except:
        raise Exception(f"{cfg.BASE_PATH}/{file}.csv not found, Enter valid one - [train/test]")


def save_file(file_name, data, cfg=cfg):
    with open(f"{cfg.BASE_PATH}/{file_name}", 'wb') as f:
        pickle.dump(data, f)

def load_file(file_name, cfg=cfg):
    with open(f"{cfg.BASE_PATH}/{file_name}", 'rb') as f:
        return pickle.load(f)