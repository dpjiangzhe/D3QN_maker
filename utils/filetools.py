import os
import io

import numpy as np
import pandas as pd
import pickle

# from utils.datatools import generate_date_list

import boto3
import gzip
from botocore.config import Config
# import s3fs

my_config = Config(
    region_name='ap-northeast-1'
)
client_s3 = boto3.client('s3', config=my_config)
bucket_name = "dp4intern"


def load_file_from_s3(fn):
    try:
        _res_one = client_s3.get_object(
            Bucket=bucket_name,
            Key=fn,
        )
    except Exception as e:
        # print("get data error:", fn, "does not exist")
        return None
    _content = _res_one['Body'].read()
    with gzip.open(io.BytesIO(_content), "rb") as f:
        _df = pd.read_csv(f, engine="python")
        return _df

    return None


def save_object_to_s3(fn, _data):
    # _data = _data_df.to_csv(engine="python", index=False)
    client_s3.put_object(Body=_data, Bucket=bucket_name, Key=fn)
    pass

"""
def add_file_to_s3(data_list, filepath, mode):
    # 往S3上的文件写入数据
    # :param data_list: 追加的数据【数据必须是字符串类型】
    # :param filepath: 数据追加到的文件【需要包含路径】
    # :param mode: 往文件写入时的写入模式
    # :return:
    bytes_to_write = data_list.encode()
    # print(bytes_to_write)
    try:
        fs = s3fs.S3FileSystem(client_kwargs={'endpoint_url': 'https://{}'.format("s3://dp4intern/test/")},
                               key=my_config['access_key'], secret=my_config['secret_key'])
        with fs.open('s3://{}/{}'.format(bucket_name, filepath), str(mode)) as f:
            f.write(bytes_to_write)
    except Exception as e:
        # print(e)
        return False
    return True
"""


def unzip_and_save_csv_file(fp, fn, columns=""):
    if not os.path.exists(fp):
        return None

    _fn = fp + fn + ".csv.gz"
    if not os.path.exists(_fn):
        return None
    try:
        if len(columns) > 0:
            _df = pd.read_csv(_fn, engine="python", names=columns, compression="gzip")
        else:
            _df = pd.read_csv(_fn, engine="python", compression="gzip")
    except Exception as e:
        return None

    _fn = fp + fn + ".csv"
    _df.to_csv(_fn, mode="w", index=False)
    print(_fn, " saved...")
    pass


def add_column_names(fn, cols=[]):
    if not os.path.exists(fn) or len(cols) == 0:
        return None
    try:
        _df = pd.read_csv(fn, engine="python", names=cols)
    except Exception as e:
        return None

    _df.to_csv(fn, mode="w", index=False)
    print(fn, " saved...")
    pass


def repack_scaler(fn):
    with open(fn, "rb") as f:
        scalar = pickle.load(f)
    print(type(scalar))
    scalars = [scalar]
    print(scalars)
    with open(fn, "wb") as f:
        pickle.dump(scalars, f)
    pass

if __name__ == "__main__":
    data_path = "../data/one_year/"
    exch = "binance"
    pair = "eth_usdt"

    """
    date_list = generate_date_list("20220913", "20220914")
    for date in date_list:
        print("process for date: ", date)
        f_path = data_path + exch + "/" + pair + date + "/"
        f_tail = "_" + exch + "_" + pair + date + "00"
        f_type = "features"
        # unzip_and_save_csv_file(f_path, f_type + f_tail)
        f_type = "tickers"
        # unzip_and_save_csv_file(f_path, f_type + f_tail)
        f_type = "trade"
        cols = ['t', 'e', 'p', 'q', 'm', 'b', 'a', 'T', 'tp']
        unzip_and_save_csv_file(f_path, f_type + f_tail, cols)
        # add_column_names(f_path+f_type+f_tail+".csv", cols)

        pass
    """

    repack_scaler("../models/202307261735/scalers.pkl")
