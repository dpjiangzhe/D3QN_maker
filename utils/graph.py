import numpy as np
import pandas as pd

import plotly
import plotly.graph_objs as go
import plotly.express as px
import plotly.io as pio


def show_account_history(acc_his: pd.DataFrame, res_file_name=""):
    _fig = px.bar(acc_his, "end_ts", "base_inv")
    if res_file_name != "":
        # 保存到文件
        plotly.offline.plot(_fig, filename=res_file_name)
    else:
        # 直接显示，需要jupyter notebook的支持
        # pio.renderers.default = 'browser'
        # pio.show(_fig)
        _fig.show()


def read_history(filename):
    _df = pd.read_csv(filename, engine="python")
    return _df


if __name__ == '__main__':
    filepath = "../data/"
    exch = "_binance_"
    trade_pair = "btc_usdt"
    date = "_2023041900"
    file_type = ".csv"

    filepath += trade_pair + date + "/"

    date_str = "202305131039"
    history_path = filepath + "history/"
    history_path += date_str + "/"

    data_type = "acc"
    file_tail = "_history"
    fn = history_path + data_type + file_tail + file_type

    print("loading account history data ...")
    hdf = read_history(fn)
    show_account_history(hdf)
