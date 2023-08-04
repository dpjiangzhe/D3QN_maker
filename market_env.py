import copy
import io
import json
import os
import pickle

import ast
import time

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import logging

import boto3
import gzip
from botocore.config import Config

import utils.action_utils as ats
import utils.datatools as dts
import utils.filetools as fts
import utils.DataManager as DM

TYPE_INT = type(0)
TYPE_STR = type("")
TYPE_FLT = type(0.0)

my_config = Config(
    region_name='ap-northeast-1'
)
client_s3 = boto3.client('s3', config=my_config)
bucket_name = "dp4intern"


# 账户信息
class AccountInfo:
    base_c = "ETH"
    quote_c = "USDC"
    base_inv = 0.0
    base_bound_upmost = 0.0    #许可持仓上限
    base_bound_limit = 0.0     #最低持仓下限
    base_portion = 0.0  #每笔订单的挂单量（暂定固定某值，可设定）
    quote_inv = 0.0
    price_prec = 0.0    #价格精度，价格变动的最小单位
    value = 0.0

    def __init__(self, config=None):
        if config is None:
            return
        for key, value in config.items():
            assert getattr(self, key) is not None
            setattr(self, key, value)

    def __str__(self):
        return "".join(str(item)+"\t" for item in (self.base_c, self.base_inv, self.base_portion, self.base_bound_upmost, self.base_bound_limit,
                                                   self.quote_c, self.quote_inv, self.price_prec))


# 账户数据
class AccountData:
    def __init__(self, acc_info: AccountInfo, inv, drawdown, peak_value, m_index):
        self.account = copy.copy(acc_info)
        self.inventory = inv
        self.drawdown = drawdown
        self.peak_value = peak_value
        self.market_index = m_index
        pass
    pass


# 订单记录
class OrderInfo:
    # 暂只考虑现货订单
    market = "SPOT"
    base_c = "ETH"
    quote_c = "USDC"

    trade_type = "BUY"
    is_taker = False
    amount = 0.0
    price = 0.0
    filled = 0.0

    upd_ts = 0

    def __init__(self, m="", base_c="", quote_c="", tt="", taker=False, amount=0.0, price=0.0, ts=0):
        if m != "":
            self.market = m
        if base_c != "":
            self.base_c = base_c
        if quote_c != "":
            self.quote_c = quote_c
        if tt != "":
            self.trade_type = tt
        self.is_taker = taker
        if amount > 0:
            self.amount = amount
        if price > 0:
            self.price = price
        if ts > 0:
            self.upd_ts = ts
        pass


def load_feature_data(fn: str, _keys=[], online=False):
    # 读取市场状态（特征）数据文件
    if online:
        _feature_df = fts.load_file_from_s3(fn)
        if _feature_df is None:
            return None
    else:
        try:
            _feature_df = pd.read_csv(fn, engine="python")
        except FileNotFoundError:
            return None

    if len(_keys) > 0:
        # self.market_df = self.market_df[_keys]
        _feature_df = dts.refine_features(_feature_df, "", _keys, True)
    _feature_df.dropna(axis=0, how="any", inplace=True)

    # self.market_df.drop("Unnamed: 0", axis=1, inplace=True)
    # print(self.market_df)
    return _feature_df


def loc_data_index(ts_list, start_ts, end_ts, index=-1):
    # 根据ts（时间戳）在ts_list中进行定位(以index为定位起点)，获取在start_ts与end_ts之间的数据条目起止索引（需明确边界ts如未与参数重合，如何选取）
    # 如某end_ts小于等于start_ts，则定位一个index，在start_ts之前离start_ts最近的数据点（含start_ts）， end_index返回None
    start_index = end_index = None
    if index > 0:
        start_index = index
    else:
        start_index = 0

    if ts_list is None:
        return None, None
    data_size = len(ts_list)
    if data_size == 0 or start_index >= data_size:
        return None, None

    n_index = d_index = start_index
    while d_index < data_size and ts_list[d_index] <= start_ts:
        n_index = d_index
        d_index += 1
    if d_index <= data_size:
        # 找到起点
        start_index = n_index
        if end_ts > start_ts:
            while d_index < data_size and ts_list[d_index] <= end_ts:
                n_index = d_index
                d_index += 1
            if d_index <= data_size:
                # 找到终点
                end_index = n_index
    else:
        start_index = None

    return start_index, end_index


def normalize_data(data, _scaler=None):
    data_n = np.asarray(data)
    data_n = np.reshape(data_n, (-1, 1))
    if _scaler is None:
        _scaler = MinMaxScaler(feature_range=(-1, 1))
        _scaler.fit(data_n)
    data_n = _scaler.transform(data_n)
    return data_n, _scaler


# 模拟市场环境，可用于线下训练及回测
class MarketEnv(object):
    action_dim = 1
    state_dim = 1
    account = AccountInfo()
    account_origin = AccountInfo()

    def __init__(self, settings):
        # , action_dim, action_edge, state_dim, tariff_rate, w_inv, w_left, w_invalid, w_cross, w_edge, log: logging.Logger=None):
        # 行动数量，状态维度，持仓惩罚系数
        # 限定只进行一个币对的交易，计价币为稳定币，只需考虑基础币的持仓

        # 保留参数设置表
        self.settings = settings

        # 行动限定为：交易价格对最优盘口价格的偏移量（基位），单位偏移量为交易币的价格精度，可正可负
        self.action_dim = self.state_dim = self.action_edge = 0
        self.tariff_rate = 0.0
        self.lamda_inv = self.lamda_left = self.lamda_invalid = 0.0
        self.lamda_cross = self.lamda_edge = 0.0

        self.change_settings(settings)

        self.inventory = 0.0
        self.order_list = []
        self.order_invalid_list = []

        # 市场数据
        self.market_df = []
        self.trade_df = self.ticker_df = self.depth_df = None
        self.market_index = []
        self.ticker_index = self.depth_index = self.trade_index = 0
        self.scalers = []
        self.index_list = {}
        self.market_info = None

        # 统计数据
        self.order_counts_buy = self.order_counts_sell = 0
        self.order_filled_buy = self.order_filled_sell = 0
        self.filled_buy_amount = self.filled_sell_amount = 0.0
        self.pair_success = self.pair_success_amount = 0.0
        self.tariff = self.tariff_quote = self.profit = self.max_drawdown = self.peak_value = 0.0
        # todo: 考虑把两种profit都按照base和quote分开记录，降低中间价格的影响
        self.total_profit = self.total_ar = self.profit_inv = self.profit_trade = 0.0
        self.turnover_rate = self.turnover_rate_per_day = 0.0
        self.timer = {"loc": 0, "get_p": 0, "store_h": 0, "other": 0}
        self.learning_loss = None
        self.evaluation = 0.0
        self.data_counter = 0

        # 过程数据记录
        self.account_history = []
        self.trade_history = []
        self.return_history = []
        self.init_time = self.clock = 0

        # 设置
        self.show_trade = False
        self.is_training = False
        self.logger = None

    def change_settings(self, settings):
        # 保留参数设置表
        self.settings = settings

        # 行动限定为：交易价格对最优盘口价格的偏移量（基位），单位偏移量为交易币的价格精度，可正可负
        # self.action_dim = action_dim
        self.action_dim = self.get_setting("action_dim", TYPE_INT)
        self.state_dim = self.get_setting("state_dim", TYPE_INT)
        self.action_edge = self.get_setting("action_edge", TYPE_INT)

        self.tariff_rate = self.get_setting("tariff_rate", TYPE_FLT)

        self.lamda_inv = self.get_setting("w_inv", TYPE_FLT)
        self.lamda_left = self.get_setting("w_left", TYPE_FLT)
        self.lamda_invalid = self.get_setting("w_invalid", TYPE_FLT)
        self.lamda_cross = self.get_setting("w_cross", TYPE_FLT)
        self.lamda_edge = self.get_setting("w_edge", TYPE_FLT)
        
        self.lamda_amount = self.get_setting('lamda_amount', TYPE_FLT)


    def reset(self, show=False, training=False, clear_history=True, data_reset=True, acc_reset=False):
        self.order_list = []
        self.order_invalid_list = []

        if acc_reset:
            self.inventory = 0.0
            self.set_account_info(self.account_origin)

            self.order_counts_buy = self.order_counts_sell = self.order_filled_buy = self.order_filled_sell = 0.0
            self.filled_buy_amount = self.filled_sell_amount = 0.0
            self.pair_success = self.pair_success_amount = 0.0
            self.tariff = self.tariff_quote = self.profit = self.max_drawdown = self.peak_value = 0.0
            self.total_profit = self.total_ar = self.profit_inv = self.profit_trade = 0.0
            self.turnover_rate = self.turnover_rate_per_day = 0.0
            self.evaluation = 0.0
            self.data_counter = 0
            self.clock = 0
        """
        else:
            # 一段数据回测结束时，准备继续之前，还原基础币持仓
            if self.market_index > 0:
                m_price, _, _ = self.get_market_price()
                self.account.quote_inv += (self.account.base_inv - self.account_origin.base_inv) * m_price
                self.account.base_inv = self.account_origin.base_inv
                self.account.value = self.account.base_inv * m_price + self.account.quote_inv
                self.inventory = self.account.base_inv
        """

        if data_reset:
            self.market_df = []
            for _item_index, _ in enumerate(self.market_index):
                self.market_index[_item_index] = 0
            self.ticker_index = self.depth_index = self.trade_index = 0
            self.market_info = None
            self.learning_loss = None

        else:
            self._get_state(0)

        if clear_history:
            self.account_history = []
            self.trade_history = []
            self.return_history = []
            self.timer = {"loc": 0, "get_p": 0, "store_h": 0, "other": 0}

        self.show_trade = show
        self.is_training = training
        return self.market_info

    def set_account_info(self, acc_info: AccountInfo):
        # 设置账户数据
        if acc_info is not None:
            self.account = copy.copy(acc_info)
            self.account_origin = copy.copy(acc_info)
            self.inventory = acc_info.base_inv

            _m_price, _, _ = self.get_market_price()
            _value = self.account_origin.base_inv * _m_price + self.account_origin.quote_inv
            self.account_origin.value = self.account.value = _value
        return

    def get_account_data(self):
        acc_data = AccountData(self.account, self.inventory, self.max_drawdown, self.peak_value, self.market_index[0])
        return acc_data

    def set_account_data(self, acc_data: AccountData = None):
        if acc_data is not None:
            self.account = copy.copy(acc_data.account)
            self.inventory = acc_data.inventory
            self.max_drawdown = acc_data.drawdown
            self.peak_value = acc_data.peak_value
            self.market_index[0] = acc_data.market_index
            if len(self.index_list) > 0:
                _index = self.get_indexes()
                self.ticker_index, self.trade_index = _index[0], _index[2]
            else:
                self.ticker_index = self.trade_index = 0
            self.calc_profit()
        pass

    def step(self, buy_action=0, sell_action=0, amount=0.0, action_ts=0, end_ts=0):
        # according to a moving window of market(暂定固定时段), change env status, and give out return value
        # 所需数据：这段时间内的回报变化（在时段结束时如仍有挂单，则撤单）

        # offline
        # 更新状态
        s0 = self.market_info
        a = [buy_action, sell_action, amount]
        self.data_counter += 1

        m_ts_list = self.market_df[0]["time"].values      # to_list()

        now_ts = time.time_ns()
        e_index, _ = loc_data_index(m_ts_list, end_ts, 0, self.market_index[0])

        if len(self.index_list) == 0:
            p_ts_list = self.ticker_df["time"].values   # to_list()
            d_s_index, d_e_index = loc_data_index(p_ts_list, action_ts, end_ts, self.ticker_index)
        else:
            _index = self.get_indexes()
            d_s_index, d_e_index = _index[0], _index[1]
        new_ts = time.time_ns()
        self.timer["loc"] += new_ts - now_ts
        now_ts = new_ts

        m_price, bid_price, ask_price = self.get_market_price(d_e_index)
        new_ts = time.time_ns()
        self.timer["get_p"] += new_ts - now_ts
        now_ts = new_ts

        # 执行行动，把订单加入order_list
        # 检查资金上限
        acc_info = self.account
        _pd = acc_info.price_prec

        buy_price = buy_amount = sell_price = sell_amount = 0.0

        if buy_action is not None:
            buy_price = bid_price + buy_action * _pd
            buy_amount = acc_info.base_portion * amount
        if sell_action is not None:
            sell_price = ask_price + sell_action * _pd
            sell_amount = acc_info.base_portion * amount
        # todo：基于一定规则确定挂单量
        # buy_amount, sell_amount = self.determine_order_amounts(buy_action, sell_action, buy_amount, sell_amount)

        r_invalid = r_cross = 0.0

        # todo: 区分处理taker和maker，目前默认是maker
        if not self.is_training:
            # 回测状态下，控制发单合法性
            if self.account.quote_inv > 0 and buy_action is not None:
                if buy_price >= ask_price:
                    buy_price = bid_price
                if buy_amount * buy_price > acc_info.quote_inv:
                    buy_amount = acc_info.quote_inv / buy_price
                if self.inventory + buy_amount > self.account.base_bound_upmost:
                    buy_amount = 0.0
                if buy_amount > 0:
                    buy_order = OrderInfo(tt="BUY", amount=buy_amount, price=buy_price, ts=action_ts)
                    self.order_list.append(buy_order)
                    self.order_counts_buy += 1
            if self.inventory < self.account.base_bound_upmost and sell_action is not None:
                # 是否要考虑持仓下限（即是否要尽量保持初始仓位）
                if sell_price <= bid_price:
                    sell_price = ask_price
                if sell_amount > acc_info.base_inv:
                    sell_amount = acc_info.base_inv
                if self.inventory - sell_amount < self.account.base_bound_limit:
                    sell_amount = 0.0
                if sell_amount > 0:
                    sell_order = OrderInfo(tt="SELL", amount=sell_amount, price=sell_price, ts=action_ts)
                    self.order_list.append(sell_order)
                    self.order_counts_sell += 1
                pass
        else:
            # 训练状态下，则把不合法订单放到order_invalid_list，在计算回报时进行特殊处理
            if buy_action is not None:
                buy_order = OrderInfo(tt="BUY", amount=buy_amount, price=buy_price, ts=action_ts)
                if self.account.quote_inv <= 0 or buy_price >= ask_price\
                        or self.inventory + buy_amount > self.account.base_bound_upmost:
                    spread = abs(buy_price - ask_price)
                    # 调整价格或未能挂单，应视为失败动作，提高惩罚
                    if spread == 0:
                        spread = self.account.price_prec
                    if buy_price >= ask_price:
                        r_cross += spread * buy_amount
                    else:
                        r_invalid += spread * buy_amount
                    self.order_invalid_list.append(buy_order)
                else:
                    self.order_list.append(buy_order)
                    self.order_counts_buy += 1
                pass
            if sell_action is not None:
                sell_order = OrderInfo(tt="SELL", amount=sell_amount, price=sell_price, ts=action_ts)
                if self.inventory < self.account.base_bound_limit or sell_price <= bid_price\
                        or self.inventory - sell_amount < self.account.base_bound_limit:
                    spread = abs(sell_price - bid_price)
                    # 调整价格或未能挂单，应视为失败动作，提高惩罚
                    if spread == 0:
                        spread = self.account.price_prec
                    if sell_price <= bid_price:
                        r_cross += spread * sell_amount
                    else:
                        r_invalid += spread * sell_amount
                    self.order_invalid_list.append(sell_order)
                else:
                    self.order_list.append(sell_order)
                    self.order_counts_sell += 1
                pass

        if e_index is not None:
            s = self._get_state(e_index)
            self.market_index[0] = e_index

            old_inv = self.inventory
            delta_r, delta_tar, delta_tar_q, inv, filled_buy, filled_sell = self.get_return_delta(action_ts, end_ts, self.trade_index)
            self.ticker_index = d_e_index

            # 统计未完成订单量
            # 未完成量对收益有惩罚项
            r_left = r_edge = 0.0
            for _order in self.order_list:
                spread = 0.0
                price_delta = 0.0
                if _order.trade_type == "BUY":
                    spread = m_price - _order.price
                    if buy_action < 0:
                        price_delta = abs(buy_action)
                else:
                    spread = _order.price - m_price
                    if sell_action > 0:
                        price_delta = sell_action
                r_left += (_order.amount - _order.filled) * abs(spread)
                r_edge += price_delta / self.action_edge
            if r_left == 0:
                r_left = -(filled_buy + filled_sell)

            r1 = self.lamda_inv * (abs(inv - self.account_origin.base_inv))
            # r1 = self.lamda_inv * abs(inv - old_inv)
            if (inv > old_inv and old_inv < self.account_origin.base_inv)\
                    or (inv < old_inv and old_inv > self.account_origin.base_inv):
                r1 = -r1
            r = delta_r - max(0, r1)
            r2 = self.lamda_left * r_left
            r -= r2
            r3 = self.lamda_invalid * r_invalid
            r -= r3
            r4 = self.lamda_cross * r_cross
            r -= r4
            r5 = self.lamda_edge * r_edge
            r -= r5
            r += delta_tar

            self.tariff += delta_tar
            self.tariff_quote += delta_tar_q
            self.calc_profit(m_price)

            new_ts = time.time_ns()
            self.timer["other"] += new_ts - now_ts
            now_ts = new_ts

            m_ts = m_ts_list[e_index]
            # print("type of m_ts: ", type(m_ts))
            history_record = {"end_index": e_index, "end_ts": end_ts, "market_ts": m_ts,
                              "return_delta": delta_r, "tariff": delta_tar,
                              "loss": self.learning_loss, "ret_value": r, "p_inv": r1, "p_left": r2, "p_invalid": r3, "p_cross": r4, "p_edge": r5}
            if not self.is_training:
                msg = f'{history_record}'
                self.log_info(msg)
            ats.store_history(self.return_history, history_record)
            history_record = {"action_ts": action_ts, "end_ts": m_ts,
                              "action_buy": a[0], "filled_buy": filled_buy, "filled_sell": filled_sell, "action_sell": a[1], "m_price": m_price,
                              "base_inv": self.account.base_inv, "quote_inv": self.account.quote_inv,
                              "t_inc": self.profit, "p_inv": self.profit_inv, "t_inv": self.profit_trade,
                              "t_profit": self.profit/self.account_origin.value, "acc_value": self.account.value,
                              "total_income": self.total_profit, "total_ar": self.total_ar,
                              "tr": self.turnover_rate, "avg_tr": self.turnover_rate_per_day,
                              "drawdown": self.max_drawdown, "tariff": self.tariff, "tariff_q": self.tariff_quote,
                              "ret_value": r, "loss": self.learning_loss}
            ats.store_history(self.account_history, history_record)

            new_ts = time.time_ns()
            self.timer["store_h"] += new_ts - now_ts
            now_ts = new_ts

            # 撤销未完成订单
            self.order_list = []
            self.order_invalid_list = []
            return s, r, a, end_ts, s0
        else:
            self.order_list = []
            self.order_invalid_list = []
            self.ticker_index = d_e_index
            return None, 0.0, a, end_ts, s0

    def _get_state(self, index):
        self.market_info, _ = self.get_market_state(index)
        return self.market_info

    def get_indexes(self, index=-1):
        # 从index_list中获取预先计算好的索引（各market数据，ticker数据和trade数据）
        if index == -1:
            index = self.market_index[0]
        _m_ts_list = self.market_df[0]["time"].values      # to_list()
        _index = self.index_list[int(_m_ts_list[index])]
        return _index

    def get_market_state(self, index=0):
        # 获取index指示的市场状态
        # 拼合多market特征数据
        s = self.market_df[0].values[index]
        ts = int(s[0])
        s = np.delete(s, 0, axis=0)
        s = np.append(s, self.inventory - self.account_origin.base_inv)
        _m_index = self.get_indexes(index)
        for _item_index, _market_df in enumerate(self.market_df):
            if _item_index == 0:
                continue
            _s = self.market_df[_item_index].values[_m_index[4][_item_index]]
            _s = np.delete(s, 0, axis=0)
            s = np.append(s, _s)
            pass
        # print(s, type(s))
        return s, ts

    def get_market_price(self, index=-1):
        # 获取当前时刻盘口价格
        if self.ticker_df is None:
            return 0.0, 0.0, 0.0

        d_index = index
        if index < 0:
            d_index = self.ticker_index
        # pb = self.ticker_df.loc[d_index, "bid"]
        pb = self.ticker_df["bid"].values[d_index]
        # pa = self.ticker_df.loc[d_index, "ask"]
        pa = self.ticker_df["ask"].values[d_index]

        m_price = (pb + pa) / 2.0

        # 调整价格符合精度要求
        n_base = int(1/self.account.price_prec)
        m_price_u = int((m_price + self.account.price_prec)*n_base) * 1.0 / n_base
        m_price_l = int(m_price * n_base) * 1.0 / n_base
        # print("market prices:", m_price, m_price_l, m_price_u)

        return m_price, m_price_l, m_price_u

    def get_return_delta(self, t1, t2, index=-1):
        # 在t1～t2之间，检查挂单成交情况（根据trade数据，模拟成交）并计算收益变化、持仓情况
        # 价差：用窗口结束时间的盘口价格与成交价格进行计算
        if t2 < t1 or t1 <= 0:
            return 0.0, 0.0, 0.0
        rd = 0.0
        inv = 0.0
        filled_buy = filled_sell = 0.0
        price_buy = price_sell = 0.0

        if len(self.index_list) == 0:
            d_ts_list = self.ticker_df["time"].values   # to_list()
            d_s_index, d_e_index = loc_data_index(d_ts_list, t1, t2, self.ticker_index)
            t_ts_list = self.trade_df["e"].values   # to_list()
            s_index, e_index = loc_data_index(t_ts_list, t1, t2, index)
        else:
            _index = self.get_indexes()
            d_s_index, d_e_index, s_index, e_index = _index[0], _index[1], _index[2], _index[3]

        m_price_s, _, _ = self.get_market_price(d_s_index)
        m_price_e, _, _ = self.get_market_price(d_e_index)
        # self.ticker_index = e_index

        if e_index is None:
            e_index = s_index
        if s_index is not None:
            # 模拟交易，成交量分别累计到filled_buy（买入挂单，对应于sell成交）和filled_sell（卖出挂单，对应于buy成交）中；挂单价格分别保存在price_和price_ask中
            while s_index <= e_index:
                df_buy, dp_buy, df_sell, dp_sell = self.match_trade_order(s_index)
                if filled_buy + df_buy > 0:
                    price_buy = (price_buy * filled_buy + dp_buy * df_buy) / (filled_buy + df_buy)
                    filled_buy = filled_buy + df_buy
                if filled_sell + df_sell > 0:
                    price_sell = (price_sell * filled_sell + dp_sell * df_sell) / (filled_sell + df_sell)
                    filled_sell = filled_sell + df_sell
                s_index += 1

            # 计算回报变动
            # 是否需要用成交时刻的市场价格与挂单（成交）价进行比较计算return变动rd
            if self.show_trade:
                history_record = {"buy": filled_buy, "price_buy": price_buy, "sell": filled_sell, "price_sell": price_sell}
                # print("[total trade matching]", "buy:", filled_buy, "with", price_buy, "\tsell:", filled_sell, "with", price_sell)
                ats.store_history(self.trade_history, history_record)

            # rd = filled_buy * (m_price_e - price_buy) + filled_sell * (price_sell - m_price_e)
            # 尝试用资产价值变动作为回报变动
            rd = (self.inventory - self.account_origin.base_inv) * (m_price_e - m_price_s) / m_price_e
            self.profit_inv += rd * m_price_e

            # 交易行为产生的收益
            # r_trade = (filled_buy - filled_sell) + (filled_sell * price_sell - filled_buy * price_buy) / m_price_e
            # r_trade = (filled_buy * (m_price_e - price_buy) + filled_sell * (price_sell - m_price_e)) / m_price_e
            # 交易匹配部分，为确定性的收益
            r_trade_f = min(filled_buy, filled_sell) * (price_sell - price_buy)
            price_t = price_buy if filled_buy > filled_sell else price_sell
            # 交易未匹配部分，形成持仓变动，收益与结束价格有关
            r_trade_u = (filled_buy - filled_sell) * (m_price_e - price_t)
            # 总计，并折算成基础币
            r_trade = (r_trade_f + r_trade_u) / m_price_e

            rd += r_trade
            self.profit_trade += r_trade * m_price_e
            # 交易费率产生的收益
            rtq = (filled_buy * price_buy + filled_sell * price_sell) * self.tariff_rate
            rt = rtq / m_price_e
            inv = self.inventory + (filled_buy - filled_sell)

            self.trade_index = e_index
            self.inventory = inv
            self.account.base_inv = inv
            self.account.quote_inv += filled_sell * price_sell - filled_buy * price_buy
            if filled_buy > 0 and filled_sell > 0:
                self.pair_success += 1
                self.pair_success_amount += min(filled_sell, filled_buy)
            pass

        return rd, rt, rtq, inv, filled_buy, filled_sell

    def match_trade_order(self, trade_index):
        # 匹配交易订单，模拟maker订单的成交情况
        filled_buy = filled_sell = 0.0
        price_buy = price_sell = 0.0

        if trade_index >= self.trade_df.shape[0]:
            return 0.0, 0.0, 0.0, 0.0

        """
        trade = self.trade_df.loc[trade_index, :]
        trade_amount = trade.loc["q"]
        trade_price = trade.loc["p"]
        trade_type = trade.loc["m"]
        trade_time = trade.loc["e"]
        """

        """"""
        # trade = self.trade_df   # .values[trade_index]
        trade_amount = self.trade_df["q"].values[trade_index]     # trade[3]
        trade_price = self.trade_df["p"].values[trade_index]     # trade[2]
        trade_type = self.trade_df["m"].values[trade_index]     # trade[4]
        trade_time = self.trade_df["e"].values[trade_index]     # trade[1]
        """"""
        for _index, order in enumerate(self.order_list):
            if order.is_taker:
                # 此处仅处理maker订单
                continue
            if order.filled < order.amount and order.trade_type != trade_type and order.upd_ts <= trade_time:
                # trade方向与maker挂单相反，可能成交
                price = order.price
                if (order.trade_type == "BUY" and price >= trade_price) \
                        or (order.trade_type == "SELL" and price <= trade_price):
                    # 找到可成交的trade
                    delta = trade_amount
                    if order.amount - order.filled < trade_amount:
                        delta = order.amount - order.filled
                    if order.trade_type == "BUY":
                        if delta > 0:
                            if order.filled == 0:
                                self.order_filled_buy += 1
                            price_buy = (filled_buy * price_buy + delta * price) / (filled_buy + delta)
                            filled_buy += delta
                            self.filled_buy_amount += delta
                    else:
                        if delta > 0:
                            if order.filled == 0:
                                self.order_filled_sell += 1
                            price_sell = (filled_sell * price_sell + delta * price) / (filled_sell + delta)
                            filled_sell += delta
                            self.filled_sell_amount += delta

                    # 根据成交情况，修改订单记录和成交记录
                    self.order_list[_index].filled = order.filled + delta

                    # if self.show_trade:
                    #    print("<<matching trader/order>>", order.trade_type, delta, '/', self.order_list[_index].amount,
                    #          " with", price, "(", trade["m"], trade_price, ")", "\tat", trade.loc["e"])
                    trade_amount -= delta
                    if trade_amount <= 0:
                        break
        # if self.show_trade:
        #    print("[total trade matching]", filled_buy, "with", price_buy, "(", trade_price, ")", "\t", filled_sell, "with", price_sell, "(", trade_price, ")")
        return filled_buy, price_buy, filled_sell, price_sell

    def match_depth_data(self, order: OrderInfo, depth_index):
        # 匹配深度数据，模拟taker订单的成交情况
        filled_buy = filled_sell = 0.0
        price_buy = price_sell = 0.0

        if depth_index >= self.depth_df.shape[0] or not order.is_taker:
            return 0.0, 0.0, 0.0, 0.0

        depth = self.depth_df   # .loc[depth_index, :]
        if order.trade_type == "BUY":
            depth_data = depth["asks"].values[depth_index]
        else:
            depth_data = depth["bids"].values[depth_index]

        price = order.price
        for _index, _data in enumerate(depth_data):
            if (order.trade_type == "BUY" and _data["p"] > price) or (order.trade_type == "SELL" and _data["p"] < price):
                break
            delta = _data["s"]
            if order.amount - order.filled < delta:
                delta = order.amount - order.filled

            if order.trade_type == "BUY":
                if delta > 0:
                    if order.filled == 0:
                        self.order_filled_buy += 1
                    price_buy = (filled_buy * price_buy + delta * _data["p"]) / (filled_buy + delta)
                    filled_buy += delta
                    self.filled_buy_amount += delta
            else:
                if delta > 0:
                    if order.filled == 0:
                        self.order_filled_sell += 1
                    price_sell = (filled_sell * price_sell + delta * _data["p"]) / (filled_sell + delta)
                    filled_sell += delta
                    self.filled_sell_amount += delta

            # if self.show_trade:
            #    print("<<matching trader/order>>", order.trade_type, delta, '/', self.order_list[_index].amount,
            #          " with", price, "(", trade["m"], trade_price, ")", "\tat", trade.loc["e"])
            order.filled += delta
            if order.amount <= order.filled:
                break
        # if self.show_trade:
        #    print("[total trade matching]", filled_buy, "with", price_buy, "(", trade_price, ")", "\t", filled_sell, "with", price_sell, "(", trade_price, ")")
        return filled_buy, price_buy, filled_sell, price_sell

    def load_market_data(self, fn: str, _keys=[], online=False):
        # 读取市场状态（特征）数据文件
        _market_df = load_feature_data(fn, _keys, online)
        self.market_df.append(_market_df)
        self.market_index.append(0)
        return _market_df is not None

    def load_depth_data(self, fn: str, online=False):
        # 读取深度数据
        self.depth_index = 0
        if online:
            self.depth_df = fts.load_file_from_s3(fn)
        elif os.path.exists(fn):
            self.depth_df = pd.read_csv(fn, engine="python")
            self.depth_df.dropna(axis=0, how="any", inplace=True)
        else:
            self.depth_df = None
        return self.depth_df is not None

    def load_ticker_data(self, fn: str, online=False):
        # 读取盘口价格数据
        self.ticker_index = 0
        if online:
            _ticker_df = fts.load_file_from_s3(fn)
            if _ticker_df is None:
                return False
        else:
            try:
                _ticker_df = pd.read_csv(fn, engine="python")
                _ticker_df.dropna(axis=0, how="any", inplace=True)
            except FileNotFoundError:
                return False

        data_list = _ticker_df["bid"].to_list()
        bid_list = []
        for item in data_list:
            if type(item) == TYPE_FLT:
                bid_list.append(item)
            else:
                item_data = ast.literal_eval(item)
                bid_list.append(item_data["p"])
        data_list = _ticker_df["ask"].to_list()
        ask_list = []
        for item in data_list:
            if type(item) == TYPE_FLT:
                ask_list.append(item)
            else:
                item_data = ast.literal_eval(item)
                ask_list.append(item_data["p"])
        _ticker_df["bid"] = bid_list
        _ticker_df["ask"] = ask_list

        self.ticker_df = _ticker_df
        # self.ticker_df.drop("Unnamed: 0", axis=1, inplace=True)
        # print(self.ticker_df)
        return True

    def load_trade_data(self, fn: str, online=False):
        # 读取交易数据文件
        self.trade_index = 0

        if online:
            self.trade_df = fts.load_file_from_s3(fn)
            if self.trade_df is None:
                return False
        else:
            try:
                self.trade_df = pd.read_csv(fn, engine="python")
                self.trade_df.dropna(axis=0, how="any", inplace=True)
            except FileNotFoundError:
                return False

        # print(self.trade_df)
        cols = ['t', 'e', 'p', 'q', 'm', 'b', 'a', 'T', 'tp']
        self.trade_df.columns = cols
        return True

    def load_index(self, fn: str, online=False):
        # 读取索引文件
        if online:
            self.index_list = fts.load_file_from_s3(fn)
            if self.index_list is None:
                self.index_list = dts.build_index(self.market_df, self.trade_df, self.ticker_df, "", True)
        else:
            try:
                with open(fn, 'r') as json_file:
                    _json_str = json_file.read()
                self.index_list = json.loads(_json_str)
            except FileNotFoundError:
                self.index_list = dts.build_index(self.market_df, self.trade_df, self.ticker_df, fn, True)
            pass

    def prepare_data(self):
        if len(self.scalers) == 0:
            self.scalers = [{}] * len(self.market_df)
        for _item_index, _market_df in enumerate(self.market_df):
            # 去除无效数据，并对除time列以外的特征数据按列进行归一化处理
            if _market_df is None:
                continue
            _market_df.dropna(axis=0, how="any", inplace=True)

            for col in _market_df.columns:
                if col == "time":
                    continue
                data_list = _market_df[col].values      # .to_list()
                scaler = None
                # 扩展scalers
                if col in self.scalers[_item_index]:
                    scaler = self.scalers[_item_index][col]
                data_list, scaler = normalize_data(data_list, scaler)
                _market_df[col] = data_list
                self.scalers[_item_index][col] = scaler
                # print(data_list, _market_df[col])
                pass
            # print(_market_df)
            pass

    def determine_order_amounts(self, buy_action, sell_action, ref_buy_amount, ref_sell_amount):
        buy_amount = ref_buy_amount
        sell_amount = ref_sell_amount
        # todo: 决定挂单量的逻辑
        # 暂用仓位加以控制
        if self.inventory > self.account_origin.base_inv * 1.05:
            buy_amount = 0.0
        elif self.inventory < self.account_origin.base_inv * 0.95:
            sell_amount = 0.0
        return buy_amount, sell_amount

    def calc_profit(self, price=0.0):
        # 指定价格，计算利润
        if price == 0:
            # 如未提供价格，则取环境中当前的市场价
            price, _, _ = self.get_market_price()
        self.profit = (self.inventory - self.account_origin.base_inv) * price
        self.profit += self.account.quote_inv - self.account_origin.quote_inv
        self.total_profit = self.profit + self.tariff_quote
        _duration = (self.clock-self.init_time)/1000/3600/24
        if _duration > 0:
            self.total_ar = (self.total_profit / _duration) *365

        # 计算资产价值，并计算最大回撤
        # note: 需确认是否应加入tariff
        _value = self.account.base_inv * price + self.tariff_quote + self.account.quote_inv
        self.account.value = _value
        if _value > self.peak_value:
            self.peak_value = _value
        _drawdown = (self.peak_value - _value) / self.peak_value
        if _drawdown > self.max_drawdown:
            self.max_drawdown = _drawdown

        # 计算模型评价估值
        self.turnover_rate = (self.tariff_quote / self.tariff_rate) / self.account_origin.value
        self.turnover_rate_per_day = self.turnover_rate / _duration
        self.evaluation = self.get_evaluation(price)
        return self.profit, price

    def get_evaluation(self, _price):
        c1 = 0.5
        c2 = 0.5
        c3 = 0.5

        if self.account_origin.base_inv > 0:
            _v = -c1 * abs(self.inventory - self.account_origin.base_inv) / self.account_origin.base_inv
        else:
            _v = 0.0
        _eval = _v
        if self.account_origin.value > 0:
            _v = c2 * self.profit / self.account_origin.value
        else:
            _v = 0.0
        _eval += _v
        if self.data_counter > 0:
            _v = c3 * (self.filled_buy_amount + self.filled_sell_amount) / self.data_counter
        else:
            _v = 0.0
        _eval += _v
        return _eval

    def log_info(self, msg):
        ats.log_info(self.logger, msg)

    def save_history(self, file_path, mod_path=""):
        # 暂定：在aws上运行（训练）不保存中间数据
        if self.is_training and self.settings['save2s3']:
            return

        # 保存过程数据
        if not os.path.exists(file_path):
            os.mkdir(file_path)

        tfn = "t_"

        # test for scalers
        """
        fn = "feature.csv"
        if self.is_training:
            fn = tfn + fn
        self.market_df.to_csv(file_path + fn)
        #####################
        """

        if len(self.account_history) > 0:
            fn = "acc_history.csv"
            if self.is_training:
                fn = tfn + fn
            history_df = pd.DataFrame(self.account_history)
            if os.path.exists(file_path+fn):
                header = False
            else:
                header = True
            history_df.to_csv(file_path + fn, mode="a", header=header, index=False)
        if len(self.return_history) > 0:
            fn = "ret_history.csv"
            if self.is_training:
                fn = tfn + fn
            history_df = pd.DataFrame(self.return_history)
            if os.path.exists(file_path+fn):
                header = False
            else:
                header = True
            history_df.to_csv(file_path + fn, mode="a", header=header, index=False)
        if len(self.trade_history) > 0:
            fn = "tra_history.csv"
            if self.is_training:
                fn = tfn + fn
            history_df = pd.DataFrame(self.trade_history)
            if os.path.exists(file_path+fn):
                header = False
            else:
                header = True
            history_df.to_csv(file_path + fn, mode="a", header=header, index=False)

        if len(self.scalers) > 0 and mod_path != "":
            if not self.settings["save2s3"] and not os.path.exists(mod_path):
                os.mkdir(mod_path)
            fn = "scalers.pkl"
            if self.is_training:
                if not self.settings["save2s3"]:
                    p_file = open(mod_path + fn, "wb")
                    pickle.dump(self.scalers, p_file)
                    p_file.close()
                else:
                    _object = pickle.dumps(self.scalers)
                    fts.save_object_to_s3(mod_path + fn, _object)

    def get_setting(self, key: str, value_type):
        # 在settings中获取指定key的设置值
        if key in self.settings and type(self.settings[key]) == value_type:
            setting_value = self.settings[key]
        else:
            if value_type == TYPE_INT:
                setting_value = 0
            elif value_type == TYPE_STR:
                setting_value = ""
            else:
                setting_value = 0.0
        return setting_value

    def load_scalers(self, file_path, fn):
        # 读取归一化模型
        p_file = open(file_path + fn, "rb")
        print(file_path+fn)
        self.scalers = pickle.load(p_file)
        print(self.scalers)
        p_file.close()


