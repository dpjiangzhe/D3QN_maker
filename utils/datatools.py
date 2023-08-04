import os

import numpy as np
import pandas as pd
import json
from datetime import datetime
from datetime import timedelta


def refine_features(_f_df, _fn, _keys, online=False):
    _data_df = _f_df.copy()

    _col_names = []
    _f_names = []
    print("refining features: ", _fn)
    for key in _keys:
        if isinstance(key, type("")):
            _col_names.append(key)
            _f_names.append(key)
        else:
            if isinstance(key, type([])):
                _col_names.append(key[0])
                for k in key:
                    if k in _data_df.columns:
                        _f_names.append(k)
                        break

    _data_df = _data_df[_f_names]
    _data_df.dropna(axis=0, how="any", inplace=True)
    # print(_data_df)
    _data_df.columns = _col_names

    if not online:
        _data_df.to_csv(_fn, index=False)
    return _data_df


def merge_features(_dest_fn, _main_df, _new_fn, _keys):
    # _main_df = pd.read_csv(_main_fn)
    _new_df = pd.read_csv(_new_fn)

    _ts_list = _new_df['time'].to_list()
    _data_df = _main_df[_main_df['time'].isin(_ts_list)].copy()
    _ts_list = _data_df['time'].to_list()
    _new_df = _new_df[_new_df['time'].isin(_ts_list)]
    for _k in _keys:
        _data_df[_k] = _new_df[_k].to_list()
    _data_df.dropna(axis=0, how="any", inplace=True)
    _data_df.to_csv(_dest_fn, index=False)


def concat_files(_fp, _pn, _exch, _date_str, _exch_list, _date_list):
    _f_data_df = None
    _tr_data_df = None
    _tk_data_df = None
    for _index, _date in enumerate(_date_list):
        _ofp = _pn + "_" + _date + "/"
        _file_tail = _exch_list[_index] + _pn + "_" + _date_list[_index] + ".csv"

        _fn = _fp + _ofp + "features" + _file_tail
        _data_df = pd.read_csv(_fn, engine="python")
        if _f_data_df is None:
            _f_data_df = _data_df
        else:
            _f_data_df = pd.concat([_f_data_df, _data_df])
            _f_data_df.reset_index(inplace=True)

        _fn = _fp + _ofp + "tickers" + _file_tail
        _data_df = pd.read_csv(_fn, engine="python")
        if _tk_data_df is None:
            _tk_data_df = _data_df
        else:
            _tk_data_df = pd.concat([_tk_data_df, _data_df])
            _tk_data_df.reset_index(inplace=True)

        _fn = _fp + _ofp + "trade" + _file_tail
        _data_df = pd.read_csv(_fn, engine="python")
        if _tr_data_df is None:
            _tr_data_df = _data_df
        else:
            _tr_data_df = pd.concat([_tr_data_df, _data_df])
            _tr_data_df.reset_index(inplace=True)

    _file_tail = _exch + _pn + "_" + _date_str + ".csv"
    _fp += _pn + "_" + _date_str + "/"
    if not os.path.exists(_fp):
        os.mkdir(_fp)
    if _f_data_df is not None:
        _fn = _fp + "features" + _file_tail
        _f_data_df.to_csv(_fn, index=False)
    if _tk_data_df is not None:
        _fn = _fp + "tickers" + _file_tail
        _tk_data_df.to_csv(_fn, index=False)
    if _tr_data_df is not None:
        _fn = _fp + "trade" + _file_tail
        _tr_data_df.to_csv(_fn, index=False)


def build_index(_m_df, _tr_df, _tk_df, _fni, online=False):
    # market_df = pd.read_csv(_fnf, engine="python")
    # trade_df = pd.read_csv(_fnt, engine="python")
    # depth_df = pd.read_csv(_fnd, engine="python")

    trade_len, ticker_len = _tr_df.shape[0], _tk_df.shape[0]

    res = {}
    ticker_start, ticker_end, trade_start, trade_end = 0, 0, 0, 0
    market_index = [0] * len(_m_df)
    print("building indexes: ", _fni)
    # for _, row in _m_df.iterrows():
    _tk_list = _tk_df["time"].values
    _tr_list = _tr_df["e"].values
    _m_list = _m_df[0]["time"].values
    data_size = len(_m_list)
    # for _, item in _m_df.iterrows():
    for i in range(data_size):
        # ts = int(row['time'])
        ts = int(_m_list[i])
        # while ticker_start + 1 < ticker_len and int(_tk_df.loc[ticker_start + 1, 'time']) <= ts:
        while ticker_start + 1 < ticker_len and int(_tk_list[ticker_start + 1]) <= ts:
            ticker_start += 1
        # while trade_start + 1 < trade_len and int(_tr_df.loc[trade_start + 1, 'e']) <= ts:
        while trade_start + 1 < trade_len and int(_tr_list[trade_start + 1]) <= ts:
            trade_start += 1
        for _item_index, _market_df in enumerate(_m_df):
            if _item_index == 0:
                continue
            _m_ts_list = _market_df["time"].values
            market_index[0] = i
            market_start = market_index[_item_index]
            while market_start + 1 < _market_df.shape[0] and int(_m_ts_list[market_start + 1]) <= ts:
                market_start += 1
            market_index[_item_index] = market_start

        ticker_end, trade_end = ticker_start + 1, trade_start + 1
        new_ts = ts + 10000
        # while ticker_end + 1 < ticker_len and int(_tk_df.loc[ticker_end + 1, 'time']) <= new_ts:
        while ticker_end + 1 < ticker_len and int(_tk_list[ticker_end + 1]) <= new_ts:
            ticker_end += 1
        # while trade_end + 1 < trade_len and int(_tr_df.loc[trade_end + 1, 'e']) <= new_ts:
        while trade_end + 1 < trade_len and int(_tr_list[trade_end + 1]) <= new_ts:
            trade_end += 1
        res[ts] = [ticker_start, ticker_end, trade_start, trade_end, market_index.copy()]

    if not online:
        with open(_fni, 'w') as out:
            out.write(json.dumps(res))

    return res


def generate_date_list(start_date: str, end_date: str):
    _date_list = []
    _start_time = datetime.strptime(start_date, "%Y%m%d")
    _end_time = datetime.strptime(end_date, "%Y%m%d")

    _cur_time = _start_time
    while _cur_time <= _end_time:
        _cur_date = "_" + _cur_time.strftime("%Y%m%d")
        _date_list.append(_cur_date)
        _cur_time = _cur_time + timedelta(days=1)

    return _date_list


if __name__ == '__main__':
    data_filepath = "../data/"
    exch = "_binance_"
    trade_pair = "eth_usdt"
    date = "_2023051607"
    file_type = ".csv"

    filepath = data_filepath + trade_pair + date + "/"
    file_tail = exch + trade_pair + date + file_type
    fn1 = filepath + "features" + file_tail
    market_df = pd.read_csv(fn1, engine="python")
    fn2 = filepath + "features_rf" + file_tail
    keys = ["time", "mid_price", ["bid_distance_10", "bids_distance_10"], ["ask_distance_10", "asks_distance_10"],
            ["rsi_15s", "RSI_15s"], ["rsi_60s", "RSI_60s"], ["crsi_15s", "CRSI_15s"], ["crsi_60s", "CRSI_60s"],
            "trade_flow_imbalances_10s", "trade_volume_5s",
            "vwap_5s",
            "cumulative_value_ask_level-10", "cumulative_value_bid_level-10", "notional_imbalances_level-10",
            "5_level_cumulative_ask", "5_level_cumulative_bid", "5_level_notional_imbalances"]
    refine_features(market_df, fn2, keys)

    """
    exch_list = ["_binance_", "_binance_", "_binance_"]
    date_list = ["2023041308", "2023042107", "2023051517"]
    concat_files(data_filepath, trade_pair, exch, "2023041372", exch_list, date_list)
    """

    """
    predict_keys = ["yi_predict", "xgboost_prediction"]
    dest_fn = filepath + "features_merge" + file_tail
    main_fn = fn2
    new_fn = filepath + "predictions" + file_tail
    merge_features(dest_fn, market_df, new_fn, predict_keys)
    """

    fn3 = filepath + "trade" + file_tail
    trade_df = pd.read_csv(fn3, engine="python")
    fn4 = filepath + "tickers" + file_tail
    tickers_df = pd.read_csv(fn4, engine="python")
    fn5 = filepath + "time_index" + file_tail
    build_index(market_df, trade_df, tickers_df, fn5)
