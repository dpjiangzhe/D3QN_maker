import pandas as pd
import numpy as np
import gzip
import json
import copy
from datetime import datetime, timedelta, timezone
import re

from ast import literal_eval
import boto3
from botocore.config import Config
from typing import Generator

DEFAULT_LEVEL = 20
DEFAULT_PACKS = 24

# file source
UNKNOWN_FILE = ""
LOCAL_FILE = "LOCAL"
S3_FILE = "S3"

# file type
CSV_FILE = "csv"
LOG_FILE = "log"

# data path struct type
PST_NORMAL = "normal"
PST_D3QN = "d3qn"

# data type scribe
DS_EMPTY = ""
DS_DEPTH = "depth"
DS_TICKERS = "tickers"
DS_TRADE = "trade"
DS_FEATURES = "features"
DS_MARKET_PRICE = "m_price"

# bucket name
BN_DEPTH = "depths"
BN_TICKER = "dpticker"
BN_TRADE = "dp-trade"
BN_INTERN = "dp4intern"


class DataConfig(object):
    prime_coin = ""                 # 基本币
    quote_coin = ""                 # 计价币
    chain = ""                      # 链名
    exchange = ""                   # 交易所名
    trade_type = ""                 # 交易类型/合约名
    date_hour = ""                  # 数据起始时间：YYYYMMDDHH
    end_date_hour = ""              # 数据结束时间：YYYYMMDDHH
    depth_level = DEFAULT_LEVEL     # 默认行情深度
    source = LOCAL_FILE             # 数据来源
    data_root = ""                  # 数据根目录
    file_type = LOG_FILE            # 数据文件类型
    compressed = True               # 是否压缩文件
    pst = PST_NORMAL                # 文件目录结构类型

    def __init__(self, prime_coin="", quote_coin="", chain="", exchange="", trade_type="", date_hour="", end_date_hour="",
                 depth_level=DEFAULT_LEVEL, source=LOCAL_FILE, data_root="", file_type=LOG_FILE, compressed=True, pst=PST_NORMAL):
        self.init_config(prime_coin, quote_coin, chain, exchange, trade_type, date_hour, end_date_hour,
                         depth_level, source, data_root, file_type, compressed, pst)
        pass

    def init_config(self, prime_coin="", quote_coin="", chain="", exchange="", trade_type="", date_hour="", end_date_hour="",
                    depth_level=DEFAULT_LEVEL, source="", data_root="", file_type=LOG_FILE, compressed=True, pst=PST_NORMAL):
        self.prime_coin = prime_coin.upper()
        self.quote_coin = quote_coin.upper()
        self.chain = chain
        self.exchange = exchange
        self.trade_type = trade_type
        self.date_hour = date_hour
        self.end_date_hour = end_date_hour
        if self.end_date_hour == "":
            self.end_date_hour = date_hour
        self.depth_level = depth_level
        self.source = source
        self.data_root = data_root
        self.file_type = file_type
        self.compressed = compressed
        self.pst = pst

    def set_config(self, config=None):
        if config is not None and len(config) > 0:
            for key, value in config.items():
                assert getattr(self, key) is not None
                setattr(self, key, value)
        pass

    def get_config(self):
        return self.__dict__

    def check(self):
        _ok = False
        _config = self.get_config()
        if len(_config) > 0:
            for key, value in _config.items():
                if key in ["source", "trade_type", "data_root", "chain", "depth_level", "file_type", "compressed", "pst"]:
                    continue
                _ok = len(value) > 0
                if not _ok:
                    break
        return _ok

    def identify(self, config):
        _eq = False

        _config = self.get_config()
        _new_config = config.get_config()
        if len(_config) > 0:
            for key, value in _config.items():
                if key in ["depth_level"]:
                    continue
                if key in _new_config:
                    _eq = value == _new_config[key]
                else:
                    _eq = False
                if not _eq:
                    break
        return _eq

    pass


class DataAttrib(object):
    prime_coin = ""                 # 基本币
    quote_coin = ""                 # 计价币
    chain = ""                      # 链名
    exchange = ""                   # 交易所名
    trade_type = ""                 # 交易类型/合约名

    def __int__(self, prime_coin="", quote_coin="", chain="", exchange="", trade_type=""):
        self.prime_coin = prime_coin.upper()
        self.quote_coin = quote_coin.upper()
        self.chain = chain
        self.exchange = exchange
        self.trade_type = trade_type

    def __str__(self):
        return "".join(str(item)+"\t" for item in (self.prime_coin, self.quote_coin, self.chain, self.exchange, self.trade_type))

    def set_attrib(self, config: DataConfig):
        self.prime_coin = config.prime_coin
        self.quote_coin = config.quote_coin
        self.chain = config.chain
        self.exchange = config.exchange
        self.trade_type = config.trade_type

    def get_attrib(self):
        return self.__dict__

    pass


class DataRecord(object):
    attrib: DataAttrib = None
    data_record_list: dict = None

    def __init__(self, config: DataConfig):
        self.attrib = DataAttrib()
        self.data_record_list = {}
        self.attrib.set_attrib(config)

    def get_attrib(self):
        return self.attrib

    def get_data(self, key=""):
        # 给定key，提取对应的数据（字典）
        # 如果key是空串，返回全部数据（字典数组）
        # 如果key不存在，返回空数组
        if key in self.data_record_list.keys():
            return self.data_record_list[key]
        elif key == "":
            return self.data_record_list
        else:
            return []

    def set_data(self, key: str, data: list):
        if data is None:
            self.data_record_list[key] = None
        else:
            self.data_record_list[key] = copy.deepcopy(data)
        return

    def merge_data(self, data=None, at_tail=True):
        if data is None:
            return

        for key in data.data_record_list.keys():
            if at_tail:
                if key in self.data_record_list.keys():
                    self.data_record_list[key] += data.data_record_list[key]
                else:
                    self.set_data(key, data.data_record_list[key])
            else:
                _data = None
                if len(data.data_record_list[key]) > 0:
                    _data = copy.deepcopy(data.data_record_list[key])
                    _data += self.data_record_list[key]
                    self.data_record_list[key] = _data
            print(f'\t\tmerge record: {key} {len(data.data_record_list[key])} {len(self.data_record_list[key])}')
            if key == DS_DEPTH:
                print(f'\t\t\t{len(data.data_record_list[key][-1]["bids"])} {len(self.data_record_list[key][-1]["bids"])}')
    pass


# 一个datapack保存单一文件记录的单一交易对及单一交易品种的全部（可用）数据，以目前的存储方式为1～n小时数据
# 包含depth、tickers、trade数据，以及可能的features数据
# 如有额外附加数据，亦可增添——数据保存在data_set字典内，键值为数据类型
class DataPack(object):
    data_set: dict = None
    data_index: dict = None
    config: DataConfig = None

    def __init__(self, config: DataConfig = None):
        self.data_set = {DS_DEPTH: None, DS_TICKERS: None, DS_MARKET_PRICE: None, DS_TRADE: None, DS_FEATURES: None}
        self.data_index = {DS_DEPTH: 0, DS_TICKERS: 0, DS_MARKET_PRICE: 0, DS_TRADE: 0, DS_FEATURES: 0}
        self.config: DataConfig = DataConfig()

        self.set_config(config)
        if self.config is not None:
            if self.config.check():
                # 全部属性皆已赋值，可直接加载数据
                self.load_data(settings)
        pass

    def set_config(self, config: DataConfig):
        if config is not None:
            self.config = copy.deepcopy(config)
        pass

    def load_data(self, config: DataConfig = None):
        # 读取常用数据：depth，tickers和trade
        # 通用S3（PST_NORMAL）路径结构：（存储桶）/YYYYMMDDHH/exchange/
        # 通用S3文件名：主币_计价币_合约.log.gz
        # 专用S3（PST_D3QN）有自己的路径结构和文件名规范
        # 如有需求，可添加自定义的路径结构和文件名规范

        _config = DataConfig()
        _config.set_config(self.config.get_config())
        if config is not None:
            if config.prime_coin != "":
                _config.prime_coin = config.prime_coin
            if config.quote_coin != "":
                _config.quote_coin = config.quote_coin
            if config.chain != "":
                _config.chain = config.chain
            if config.exchange != "":
                _config.exchange = config.exchange
            if config.trade_type != "":
                _config.trade_type = config.trade_type
            if config.date_hour != "":
                _config.date_hour = config.date_hour
            if config.end_date_hour != "":
                _config.end_date_hour = config.end_date_hour
            if config.source != "":
                _config.source = config.source
            _config.file_type = config.file_type
            _config.compressed = config.compressed
            _config.pst = config.pst

        _is_succeeded = False
        if _config.source == S3_FILE or _config.source == LOCAL_FILE:
            _is_succeeded = self.load_depth_data(_config)
            if _is_succeeded:
                # 因没有交易所时间，不能保证时序的一致性，tickers数据可能无法直接使用
                self.load_tickers_data(_config)
                self.load_trade_data(_config)
            pass
        else:
            # unknown source
            pass

        if _is_succeeded:
            self.config = _config
            pass

        return _is_succeeded

    def load_depth_data(self, config: DataConfig = None):
        if config is None:
            config = self.config

        _df = None
        # 读入depth数据
        _fpath, _fn = self.get_data_path(DS_DEPTH, config)

        if config.source == LOCAL_FILE:
            # 本地文件
            pass
        elif config.source == S3_FILE:
            if config.pst == PST_NORMAL:
                # S3文件——todo: 恢复全量depth
                _df = load_data_source_from_s3(_fpath+_fn, BN_DEPTH, config.file_type, config.compressed)
            elif config.pst == PST_D3QN:
                _df = load_data_source_from_s3(_fpath+_fn, BN_INTERN, config.file_type, config.compressed)
            else:
                pass

            pass
        else:
            # 未知来源
            pass

        if _df is not None:
            self.data_set[DS_DEPTH] = _df
            self.data_index[DS_DEPTH] = 0
            self.config.end_date_hour = get_date_str(self.data_set[DS_DEPTH][-1]["e"])
            print("data pack: depth loaded:", _fpath, _fn, len(_df), hex(id(_df)))

        return _df is not None

    def recover_depth(self, need_loading=False, config: DataConfig = None, snapshot=None):
        # 恢复全量depth；需要前一时段数据snapshot为起点，如无则以本时段第一个全量数据为起点
        # 在load_depth_data之后，data_set["depth"]数据恢复成全量depth
        if config is None:
            config = self.config

        _df = self.data_set[DS_DEPTH]
        if need_loading:
            _ok = self.load_depth_data(config)
            if _ok:
                _df = self.data_set[DS_DEPTH]

        if _df is not None and 'e' not in _df[0]:
            return False

        _depth_level = config.depth_level
        if _df is not None:
            # todo: 根据_df和snapshot恢复全量depth
            sorted_list = sorted(_df, key=lambda x: x['e'])

            _snapshot = None
            if snapshot is not None:
                _snapshot = copy.deepcopy(snapshot)

            start = 0
            res = []
            depth_update = {'bids': [], 'asks': []}
            # snapshot为空或者是未来数据，表示当前是第一个部分数据，要先跳过数据最前面不是全量的部分，找到第一条全量推送
            # 否则，snapshot传入的是上一个小时最后一个时刻的depth恢复全量的数据
            if _snapshot is None or _snapshot["time"] > sorted_list[0]['e']:
                while start < len(sorted_list) and '_' not in sorted_list[start]:
                    start += 1
                if start < len(sorted_list):
                    _snapshot = {'bids': [], 'asks': []}
            depth_time = 0

            for i in range(start, len(sorted_list)):
                data_one = sorted_list[i]
                if data_one['e'] != depth_time:
                    if depth_update['asks'] and depth_update['bids'] and '_' in depth_update['asks'][-1] and '_' in \
                            depth_update['bids'][-1]:
                        ask_index = bid_index = 0
                        while '_' not in depth_update['asks'][ask_index]:
                            ask_index += 1
                        while '_' not in depth_update['bids'][bid_index]:
                            bid_index += 1
                        _snapshot = {'bids': depth_update['bids'][bid_index:],
                                     'asks': depth_update['asks'][ask_index:],
                                     'time': data_one['e']}
                    else:
                        update_bids(depth_update['bids'], _snapshot['bids'])
                        update_asks(depth_update['asks'], _snapshot['asks'])

                    if depth_time > 0 and (_snapshot['asks'] or _snapshot['bids']):
                        _data_size_bids = min(_depth_level, len(_snapshot['bids']))
                        _data_size_asks = min(_depth_level, len(_snapshot['asks']))
                        one = {'bids': copy.deepcopy(_snapshot['bids'][:_data_size_bids]),
                               'asks': copy.deepcopy(_snapshot['asks'][:_data_size_asks]),
                               'time': depth_time}
                        res.append(one)
                        # print(f'\trecover depth: {i} {depth_time} {len(one["bids"])}')

                    if i % 500 == 0:
                        print("recovering:", i, len(res), depth_time)
                    depth_time = data_one['e']
                    depth_update = {'bids': [], 'asks': []}
                update_depth(depth_update, data_one)

            if _snapshot is not None:
                update_bids(depth_update['bids'], _snapshot['bids'])
                update_asks(depth_update['asks'], _snapshot['asks'])
                _data_size_bids = min(_depth_level, len(_snapshot['bids']))
                _data_size_asks = min(_depth_level, len(_snapshot['asks']))
                one = {'bids': copy.deepcopy(_snapshot['bids'][:_data_size_bids]),
                       'asks': copy.deepcopy(_snapshot['asks'][:_data_size_asks]),
                       'time': depth_time}
                res.append(one)
            _df = res
            pass

        if _df is not None:
            self.data_set[DS_DEPTH] = _df
            self.data_index[DS_DEPTH] = 0
            self.config.end_date_hour = get_date_str(self.data_set[DS_DEPTH][-1]["time"])
            self.generate_market_price()
            print(f'\tfinish depth recovery for {config.date_hour}')
        return _df is not None

    def generate_market_price(self):
        # 根据depth数据生成
        if self.data_set[DS_DEPTH] is None or len(self.data_set[DS_DEPTH]) == 0:
            return

        if 'time' not in self.data_set[DS_DEPTH][0]:
            return

        self.data_set[DS_MARKET_PRICE] = get_market_price_list(self.data_set[DS_DEPTH])
        self.data_index[DS_MARKET_PRICE] = 0
        print("\tgenerate market prices:", len(self.data_set[DS_MARKET_PRICE]))

    def load_tickers_data(self, config: DataConfig = None):
        # tickers数据没有交易所时间，只有保存时间tp
        # 通常情况应使用depth数据生成盘口数据
        if config is None:
            config = self.config

        _df = None
        # 读入ticker数据
        _source = config.source
        _fpath, _fn = self.get_data_path(DS_TICKERS, config)

        if _source == LOCAL_FILE:
            # 本地文件
            pass
        elif _source == S3_FILE:
            if config.pst == PST_NORMAL:
                _df = load_data_source_from_s3(_fpath+_fn, BN_TICKER, config.file_type, config.compressed)
            elif config.pst == PST_D3QN:
                _df = load_data_source_from_s3(_fpath+_fn, BN_INTERN, config.file_type, config.compressed)
            else:
                pass
            pass
        else:
            # 未知来源
            pass

        if _df is not None:
            for item in _df:
                if 'e' in item and 'tp' in item and item['e'] == 0:
                    item['e'] = item['tp']
            self.data_set[DS_TICKERS] = _df
            self.data_index[DS_TICKERS] = 0
            print("data pack: tickers loaded:", _fpath, _fn, len(_df), hex(id(_df)))

        return _df is not None

    def load_trade_data(self, config: DataConfig = None):
        if config is None:
            config = self.config

        _df = None
        # todo: 读入trade数据
        _fpath, _fn = self.get_data_path(DS_TRADE, config)

        if config.source == LOCAL_FILE:
            # 本地文件
            pass
        elif config.source == S3_FILE:
            if config.pst == PST_NORMAL:
                _df = load_data_source_from_s3(_fpath + _fn, BN_TRADE, config.file_type, config.compressed)
            elif config.pst == PST_D3QN:
                _df = load_data_source_from_s3(_fpath + _fn, BN_INTERN, config.file_type, config.compressed)
            else:
                pass
            pass
        else:
            # 未知来源
            pass

        if _df is not None:
            self.data_set[DS_TRADE] = _df
            self.data_index[DS_TRADE] = 0
            print("data pack: trades loaded:", _fpath, _fn, len(_df), hex(id(_df)))

        return _df is not None

    def load_extra_data(self, bucket=BN_INTERN, scribe=DS_FEATURES, config: DataConfig = None):
        if config is None:
            config = self.config

        _df = None
        # todo: 读取附加数据
        _fpath, _fn = self.get_data_path(scribe, config.get_config())

        if config.source == LOCAL_FILE:
            # 本地文件
            pass
        elif config.source == S3_FILE:
            if config.pst == PST_NORMAL:
                _df = load_data_source_from_s3(_fpath + _fn, bucket, config.file_type, config.compressed)
            elif config.pst == PST_D3QN:
                _df = load_data_source_from_s3(_fpath + _fn, bucket, config.file_type, config.compressed)
            else:
                pass
            pass
        else:
            # 未知来源
            pass

        # 以scribe为数据名称（键值）data_set，如果已存在则替换
        if _df is not None:
            self.data_set[scribe] = _df
            self.data_index[scribe] = 0
            print(f'data pack: {scribe} loaded: {_fn} {_fpath} {len(_df)}')

        return _df is not None

    def get_data_path(self, scribe=DS_EMPTY, config: DataConfig = None):
        _fn = _fpath = ""
        _config = copy.deepcopy(config)
        if _config is None:
            _config = self.config

        _pc = _config.prime_coin.lower()
        _qc = _config.quote_coin.lower()
        symbol = f'{_pc}_{_qc}'
        if _config.trade_type != "" and _config.trade_type != "spot":
            symbol += "_" + _config.trade_type
        postfix = _config.file_type
        if _config.compressed:
            postfix += ".gz"

        _root = _config.data_root
        if _root != "":
            _root += "/"
        if _config.pst == PST_NORMAL:
            _fn = f'{symbol}.{postfix}'
            _exchange = _config.exchange
            if _config.chain != "":
                # dex
                _exchange = f'dex_{_config.chain}_{_exchange}'
            _fpath = f'{_root}{_config.date_hour}/{_exchange}/'
            pass
        elif _config.pst == PST_D3QN:
            _fn = f'{scribe}_{symbol}.{postfix}'
            _fpath = f'{_root}{_config.exchange}/{symbol}_{_config.date_hour}/'
            pass
        else:
            pass

        return _fpath, _fn

    def get_data_by_index(self, start_index: int, end_index: int = None, data_type=DS_DEPTH):
        # 根据起止index获取对应key的数据
        if start_index < 0:
            start_index = 0
        if end_index is None:
            end_index = start_index
        if end_index >= len(self.data_set[data_type]):
            end_index = len(self.data_set[data_type]) - 1
        if end_index < start_index:
            end_index = start_index
        return self.data_set[data_type][start_index: end_index+1]

    def get_data(self, start_ts: int = None, end_ts: int = None):
        # todo：根据timestamp查找最接近的数据，默认是时间不晚于tp的
        # 如果起始时间是None，则从头取；如果同时end_ts也是None，则取全部数据
        _data_record_list = DataRecord(self.config)

        _status = 0
        for key in self.data_set.keys():
            _data_list = self.data_set[key]
            if _data_list is None:
                continue
            # print("data pack: get data:", key, len(_data_list), hex(id(_data_list)))
            # if key == DS_DEPTH:
            #   print(f'\tdepth: {len(_data_list)} {len(_data_list[0]["bids"])} at {_data_list[0]["time"]}')

            if start_ts is None:
                start_index = 0
                if end_ts is None:
                    end_index = len(_data_list)
                else:
                    end_index, _ = loc_data_index(_data_list, end_ts, self.data_index[key])
            else:
                if end_ts < 0:
                    start_index, _ = loc_data_index(_data_list, start_ts, None, self.data_index[key])
                    end_index = len(_data_list)
                else:
                    start_index, end_index = loc_data_index(_data_list, start_ts, end_ts, self.data_index[key])
            # todo: 根据index的情况截取数据
            if key == DS_DEPTH:
                if start_index == -1:
                    _status = -1
                elif start_index is None:
                    _status = 1
            if start_index is not None:
                if end_index is None:
                    end_index = start_index
                _data_record_list.set_data(key, self.get_data_by_index(start_index, end_index))
                self.data_index[key] = end_index
            else:
                _data_record_list.set_data(key, [])
            pass
        return _data_record_list, _status

    pass


########################################################################
# 一些工具函数

my_config = Config(
    region_name='ap-northeast-1'
)
client_s3 = boto3.client('s3', config=my_config)


def load_data_source_from_s3(fn, bucket=BN_DEPTH, file_type=CSV_FILE, compressed=True):
    try:
        _res_one = client_s3.get_object(
            Bucket=bucket,
            Key=fn,
        )
    except Exception as e:
        # print("get data error:", fn, "does not exist")
        return None
    _content = _res_one['Body'].read()
    _df = None

    if compressed:
        _ct_dzip = gzip.decompress(_content).decode()
        if file_type == CSV_FILE:
            pass
        elif file_type == LOG_FILE:
            record_list = _ct_dzip.split('\n')
            _df = get_data_list(record_list)
            pass

        """
        with gzip.open(io.BytesIO(_content), "rb") as f:
            if file_type == "csv":
                _df = pd.read_csv(f, engine="python")
            elif file_type == "log":
                pass
            return _df
        """
    else:
        # todo: 无须解压缩
        pass

    return _df


def get_data_list(record_list):
    """
        将结果从str转变为list
    """
    return [json.loads(row) for row in record_list if row != '']


def update_bids(res, bids_p):
    """
    用res更新bids_p
    """
    for i in res:
        bid_price = i['p']
        # 测试使用二分查找的运行效率
        l, r = 0, len(bids_p) - 1
        while l <= r:
            mid = (l + r) // 2
            if bids_p[mid]['p'] == bid_price:
                if i['s'] == 0:     #p一致，s=0，则去掉depths中的这条数据
                    del bids_p[mid]
                else:       #p一致，s！=0，则用depth——update的这条数据替换depths中的数据
                    bids_p[mid]['s'] = i['s']
                break
            if bids_p[mid]['p'] > bid_price:
                l = mid + 1
            else:
                r = mid - 1
        if l > r and i['s'] != 0:
            bids_p.insert(l, i)


def update_asks(res, asks_p):
    """
    用res更新bids_p
    """
    for i in res:
        ask_price = i['p']
        # 测试使用二分查找的运行效率
        l, r = 0, len(asks_p) - 1
        while l <= r:
            mid = (l + r) // 2
            if asks_p[mid]['p'] == ask_price:
                if i['s'] == 0:     #p一致，s=0，则去掉depths中的这条数据
                    del asks_p[mid]
                else:       #p一致，s！=0，则用depth——update的这条数据替换depths中的数据
                    asks_p[mid]['s'] = i['s']
                break
            if asks_p[mid]['p'] < ask_price:
                l = mid + 1
            else:
                r = mid - 1
        if l > r and i['s'] != 0:
            asks_p.insert(l, i)


def update_depth(depth_update, data_one):
    """
    将新数据（data_one)添加到depth_update中，准备用于更新depth
    """
    del data_one['e']
    del data_one['tp']
    if data_one['t'] == 'buy':
        depth_update['bids'].append(data_one)
    else:
        depth_update['asks'].append(data_one)
    del data_one['t']


def get_market_price_list(depth_list):
    ticker_list = []
    for item in depth_list:
        _ticker = {'ap': item["asks"][0]["p"], 'aa': item["asks"][0]["s"],
                   'bp': item["bids"][0]["p"], 'ba': item["bids"][0]["s"],
                   'time': item["time"]}
        ticker_list.append(_ticker)
    return ticker_list


def get_snapshot(dpack: DataPack):
    _snapshot = None
    if dpack is not None:
        _data_list = dpack.data_set[DS_DEPTH]
        if _data_list is not None and len(_data_list) > 0 and 'time' in _data_list[0]:
            _snapshot = _data_list[-1]
            print(f'\tsnapshot: {len(_snapshot["bids"]), hex(id(_snapshot)), hex(id(_data_list[len(_data_list)-1]))}')
    return _snapshot


def loc_data_index(data_list, start_ts, end_ts=None, index=-1, before=True):
    # 根据ts（时间戳）在ts_list中进行定位(以index为定位起点)，获取在start_ts与end_ts之间的数据条目起止索引（需明确边界ts如未与参数重合，如何选取）
    # 默认before=True，起点选取<=ts的位置；否则起点选取>=ts的位置
    # 如某end_ts小于等于start_ts，则定位一个index，在start_ts之前离start_ts最近的数据点（含start_ts）， end_index返回None
    # 返回值：(-1, None)所需数据在更早时段；(-1，end)起点在更早时段；(start, None)确定单一位置；
    #       (start, end)确定区间位置；(None, None)所需数据在更晚时段，或未找到时间数据
    end_index = None
    if index > 0:
        start_index = index
    else:
        start_index = 0

    if data_list is None:
        return None, None
    _data_size = len(data_list)
    if _data_size == 0 or start_index >= _data_size:
        return None, None

    ts_key = 'e'
    if ts_key not in data_list[0]:
        ts_key = 'time'
    if ts_key not in data_list[0]:
        ts_key = 'tp'
    if ts_key not in data_list[0]:
        return None, None

    n_index = d_index = start_index
    while d_index >= 0 and data_list[d_index][ts_key] > start_ts:
        n_index = d_index
        d_index -= 1
    if d_index < 0:
        # 起始时刻在此段数据之前
        start_index = -1
    #    return start_index, None
    else:
        while d_index < _data_size and data_list[d_index][ts_key] <= start_ts:
            n_index = d_index
            d_index += 1

        start_index = n_index
        if not before:
            start_index = d_index
        if start_index >= _data_size:
            # 起始时刻在此段数据之后
            return None, None

    # 找到起点
    if end_ts is not None and end_ts > start_ts:
        while d_index < _data_size and data_list[d_index][ts_key] <= end_ts:
            n_index = d_index
            d_index += 1
        # 终点
        end_index = n_index
        # todo：到数据段末尾可能缩减部分数据

    print("\tloc index:", start_index, end_index, start_ts, end_ts, index)
    return start_index, end_index


def get_date_str(tp: int):
    # fromtimestamp参数的单位为秒
    tp = tp/1000
    d = datetime.fromtimestamp(tp, timezone(timedelta(hours=8)))
    dt = d.strftime("%Y%m%d%H")
    return dt


def get_date_tp(date_hour: str):
    try:
        # Determine the format of the input datetime string
        if re.match(r'\d{4}-\d{2}-\d{2}-\d{2}', date_hour):
            format_str = '%Y-%m-%d-%H'
        elif re.match(r'\d{10}', date_hour):
            format_str = '%Y%m%d%H'
        else:
            raise ValueError("Invalid datetime string format")

        # Parse the input datetime string
        d = datetime.strptime(date_hour, format_str)
        d = d - timedelta(hours=8)
        tp = int(d.timestamp()) * 1000

        return tp, ""

    except ValueError as e:
        # Handle invalid datetime strings
        return 0, str(e)

    pass


def get_next_date_hours(date_hour: str, hour_duration: int = 1):
    """
    Input: '2023-04-10-03' or '2023041003', -24
    Output: '2023-04-09-03' or '2023040903'
    """

    try:
        # Determine the format of the input datetime string
        if re.match(r'\d{4}-\d{2}-\d{2}-\d{2}', date_hour):
            format_str = '%Y-%m-%d-%H'
        elif re.match(r'\d{10}', date_hour):
            format_str = '%Y%m%d%H'
        else:
            raise ValueError("Invalid datetime string format")

        # Parse the input datetime string
        dt = datetime.strptime(date_hour, format_str)

        # Calculate the timedelta for x hours
        delta = timedelta(hours=hour_duration)

        # Add the timedelta to the datetime
        new_dt = dt + delta

        # Format the new datetime as a string
        new_date_hour = new_dt.strftime(format_str)

        # print("next date hour:", new_date_hour, date_hour, hour_duration)
        return new_date_hour

    except ValueError as e:
        # Handle invalid datetime strings
        return str(e)


# 一些工具函数
#######################################################################################


# 一个DataQueue保存同一种类数据多个datapack的时间序列
class DataQueue(object):
    config: DataConfig = None
    max_amount = DEFAULT_PACKS
    data_queue: [DataPack] = None
    # data_queue = []
    queue_index = 0
    current_date_hour = ""

    def __init__(self, config: DataConfig = None):
        self.config = DataConfig()
        self.max_amount = DEFAULT_PACKS
        self.data_queue = [DataPack] * 0
        if config is not None:
            self.config = copy.deepcopy(config)
            self.current_date_hour = config.date_hour
            self.queue_index = -1

    def set_config(self, config: DataConfig = None):
        if settings is not None:
            self.config = copy.deepcopy(config)
            self.current_date_hour = config.date_hour
            self.queue_index = -1

    def loc_datapack(self, date_hour: str, continuous=False):
        p_index = 0
        found = False
        _old_date_hour = "00000000"
        if re.match(r'\d{10}', date_hour):
            if int(self.config.date_hour) <= int(date_hour) <= int(self.config.end_date_hour):
                while not found and p_index < len(self.data_queue):
                    _old_date_hour = self.data_queue[p_index].config.date_hour
                    if date_hour == _old_date_hour:
                        found = True
                    else:
                        if int(date_hour) > int(_old_date_hour):
                            p_index += 1
                        else:
                            break

                    # p_index指向新加载数据应插入的位置，或已加载对应数据的位置
                    print("loc datapack:", found, p_index, date_hour, _old_date_hour)
                if not found:
                    # 需要加载数据
                    _snapshot = None
                    _old_pack = None
                    _pre_date_hour = get_next_date_hours(date_hour, -1)
                    _next_date_hour = get_next_date_hours(date_hour)
                    print("loc and load datapack:", date_hour, _pre_date_hour, _next_date_hour)

                    # 把p_index调整到前一个位置，考察是否能拿到恢复全量depth所需snapshot
                    p_index -= 1
                    if p_index >= 0 and continuous:
                        # 如指定continuous，需要填补中间的数据（通常这是针对end_ts的处理）
                        _snapshot = None
                        while int(_old_date_hour) < int(_pre_date_hour):
                            # print("data queue: before get snapshot: ", len(self.data_queue[-1].data_set[DS_DEPTH][-1]["bids"]))
                            _snapshot = self.get_snapshot(p_index)
                            _old_date_hour = get_next_date_hours(_old_date_hour)
                            # print("data queue: after get snapshot: ", len(self.data_queue[-1].data_set[DS_DEPTH][-1]["bids"]))
                            _old_pack = self.load_datapack_at_date_hour(_old_date_hour, _snapshot)
                            # print("data queue: loading a new pack:", p_index, hex(id(_old_pack)), len(self.data_queue), len(_old_pack.data_set[DS_DEPTH]), len(self.data_queue[-1].data_set[DS_DEPTH][-1]["bids"]))
                            if _old_pack is not None:
                                p_index = self.add_new_data_pack(_old_pack, p_index+1)
                            print("data queue: after adding a new pack:", p_index, len(self.data_queue), len(self.data_queue[p_index].data_set[DS_DEPTH]), len(self.data_queue[p_index].data_set[DS_DEPTH][-1]['bids']))

                    if _old_date_hour != _pre_date_hour:
                        # 之前的数据与待插入数据不连续，则丢弃已有数据
                        p_index = -1

                    if p_index < 0:
                        # 表头
                        _old_pack = self.load_datapack_at_date_hour(_pre_date_hour)
                        if _old_pack is not None:
                            _snapshot = get_snapshot(_old_pack)
                        print(
                            f'\tloc datapack: get snapshot A: {_pre_date_hour} {p_index} {hex(id(_snapshot))} {len(_snapshot["bids"])}')
                    else:
                        _snapshot = self.get_snapshot(p_index)
                        print(
                            f'\tloc datapack: get snapshot B: {_pre_date_hour} {p_index} {hex(id(_snapshot))} {len(_snapshot["bids"])}')
                        # 获取snapshot之后，把p_index调整到应插入位置
                        p_index += 1
                        pass

                    _data_pack = self.load_datapack_at_date_hour(date_hour, _snapshot)
                    if _data_pack is not None:
                        p_index = self.add_new_data_pack(_data_pack, p_index)
                        found = True

                    pass
            else:
                p_index = None
            pass
        else:
            p_index = None
        return p_index, found

    def get_snapshot(self, pack_index=-1):
        _snapshot = None
        if len(self.data_queue) > 0 and pack_index < len(self.data_queue):
            _data_pack = self.data_queue[pack_index]
            _snapshot = get_snapshot(_data_pack)
        return _snapshot

    def load_datapack_at_date_hour(self, date_hour: str, snapshot=None):
        if re.match(r'\d{10}', date_hour):
            _config = DataConfig()
            _config.set_config(self.config.get_config())
            _config.date_hour = date_hour

            _data_pack = DataPack()
            _data_pack.set_config(_config)
            ok = _data_pack.load_data()
            print("data queue: prepare data:", _config.date_hour, _config.get_config(), ok)
            # if len(self.data_queue) > 0:
            #    print(f'\tqueue: {len(self.data_queue[-1].data_set[DS_DEPTH][-1]["bids"])}')
            if ok:
                if DS_DEPTH in _data_pack.data_set.keys():
                    # 恢复depth全量数据
                    _data_pack.recover_depth(snapshot=snapshot)

                    """
                    if snapshot is not None and len(self.data_queue) > 0:
                        print("data queue: depth recovered A:", _config.date_hour, snapshot["asks"][0], snapshot["bids"][0],
                            snapshot["time"], len(_data_pack.data_set[DS_DEPTH][-1]['bids']), len(self.data_queue[-1].data_set[DS_DEPTH][-1]["bids"]))
                    else:
                        print("data queue: depth recovered B:", _config.date_hour, len(self.data_queue),
                              hex(id(_data_pack.data_set[DS_DEPTH][-1])), len(_data_pack.data_set[DS_DEPTH][-1]['bids']), hex(id(snapshot)), snapshot)
                    """

                    return _data_pack
        return None

    def add_new_data_pack(self, data_pack: DataPack, q_index=-1):
        # 把新的datapack保存到q_index指定的位置
        # q_index: -1 与已有数据不相连；0 插入到表头；>=maxamount 去头，插入表尾
        #           其它 插入，作为表尾
        if data_pack is None:
            return None
        _data_pack = copy.deepcopy(data_pack)

        if q_index < 0:
            self.data_queue = [_data_pack]
            self.queue_index = 0
            if len(self.data_queue) > 0:
                print("\t after adding new pack:", len(self.data_queue), len(self.data_queue[-1].data_set[DS_DEPTH][-1]["bids"]))
        elif q_index == 0:
            if len(self.data_queue) > 0:
                print("\t before adding new pack:", len(self.data_queue), len(self.data_queue[-1].data_set[DS_DEPTH][-1]["bids"]))
                print("\t new data pack:", len(_data_pack.data_set[DS_DEPTH][-1]["bids"]))
            q_end = len(self.data_queue)
            if q_end >= self.max_amount:
                q_end -= 1
            old_queue = self.data_queue[0:q_end]
            self.data_queue = [_data_pack]
            self.data_queue = self.data_queue + old_queue
            self.queue_index = q_end
            if len(self.data_queue) > 0:
                print("\t after adding new pack:", len(self.data_queue), len(self.data_queue[-1].data_set[DS_DEPTH][-1]["bids"]))
        elif q_index >= self.max_amount:
            self.data_queue = self.data_queue[1:]
            self.data_queue.append(_data_pack)
            self.queue_index = 0
        else:
            if len(self.data_queue) > 0:
                print("\t before adding new pack:", len(self.data_queue), len(self.data_queue[-1].data_set[DS_DEPTH][-1]["bids"]))
                print("\t new data pack:", len(_data_pack.data_set[DS_DEPTH][-1]["bids"]))
            self.data_queue = self.data_queue[0:q_index]
            self.data_queue.append(_data_pack)
            self.queue_index = q_index
            if len(self.data_queue) > 0:
                print("\t after adding new pack:", len(self.data_queue), len(self.data_queue[-1].data_set[DS_DEPTH][-1]["bids"]))

        self.show_data_queue()

        print("data queue: add new pack:", len(self.data_queue), q_index, self.queue_index, _data_pack.config.date_hour, "\n")
        return self.queue_index

    def show_data_queue(self):
        if self.data_queue is not None and len(self.data_queue) > 0:
            print(f'\t\tdata queue: {self.config.date_hour}~{self.config.end_date_hour}')
            for index, dpack in enumerate(self.data_queue):
                print(f'\t\tdata pack: {index}/{len(self.data_queue)} {dpack.config.date_hour} in '
                      f'{hex(id(dpack))}({hex(id(dpack.data_set))}, {hex(id(dpack.data_set[DS_DEPTH]))}, {hex(id(dpack.data_set[DS_TICKERS]))}, {hex(id(dpack.data_set[DS_TRADE]))})\t{len(dpack.data_set[DS_DEPTH])} {len(dpack.data_set[DS_DEPTH][-1]["bids"])}')
        pass

    def get_data(self, start_ts: int, end_ts: int = None):
        # todo: 根据ts，获取对应时刻（附近）的数据
        # 如果end_ts为None或小于等于start_ts，则只取一条数据

        # 注意：end_ts - start_ts应在一小时以内；目前似只有maker模型（强化学习）策略需要取一段数据
        # if end_ts is not None and end_ts - start_ts >= 1000 * 3600:
        #    end_ts = start_ts + 1000 * 3600

        _data_record = _pre_data = _tail_data = None
        _tail_ts = _tail_index = 0
        _status = 0

        _date_hour = get_date_str(start_ts)
        p_index, _ok = self.loc_datapack(_date_hour)
        print("get data: ready for:", _date_hour, p_index, _ok, start_ts, end_ts)

        if _ok:
            _data_pack = self.data_queue[p_index]
            if end_ts is None or end_ts < start_ts:
                end_ts = start_ts

            # 以depth数据为基准，按照时间戳定位数据
            start_index, end_index = loc_data_index(_data_pack.data_set[DS_DEPTH], start_ts, end_ts, _data_pack.data_index[DS_DEPTH])
            if start_index == -1:
                # 截取上一datapack的末尾数据作为起始
                if p_index > 0:
                    _pre_datapack = self.data_queue[p_index - 1]
                    _ts = _pre_datapack.data_set[DS_DEPTH][-1]["time"]
                    _pre_data, _status = _pre_datapack.get_data(_ts)
            elif start_index is None:
                # 截取本datapack的末尾数据作为起始
                _pre_datapack = self.data_queue[p_index]
                _ts = _pre_datapack.data_set[DS_DEPTH][-1]["time"]
                _pre_data, _status = _pre_datapack.get_data(_ts)
            else:
                _data_pack.data_index[DS_DEPTH] = start_index
                _data_pack.data_index[DS_MARKET_PRICE] = start_index
                _data_record, _status = _data_pack.get_data(start_ts, end_ts)

            print(f'\tdata record 0: {len(_data_record.data_record_list[DS_DEPTH])}\t{len(_data_record.data_record_list[DS_DEPTH][-1]["bids"])} at {get_date_str(_data_record.data_record_list[DS_DEPTH][-1]["time"])}')
            _data_record.merge_data(_pre_data, at_tail=False)
            print(f'\tdata record 1: {len(_data_record.data_record_list[DS_DEPTH])}\t{len(_data_record.data_record_list[DS_DEPTH][-1]["bids"])} at {get_date_str(_data_record.data_record_list[DS_DEPTH][-1]["time"])}')
            if end_ts is not None:
                _end_date_hour = get_date_str(end_ts)
                t_index, _ok = self.loc_datapack(_end_date_hour, continuous=True)
                if _ok and t_index > p_index:
                    while p_index + 1 < t_index:
                        p_index += 1
                        _data_pack = self.data_queue[p_index]
                        _record, _status = _data_pack.get_data()
                        _data_record.merge_data(_record)
                        print(
                            f'\tdata record 2: {p_index} {hex(id(_data_pack))} {len(_data_pack.data_set[DS_DEPTH][-1]["bids"])} {len(_record.data_record_list[DS_DEPTH])}\t{len(_record.data_record_list[DS_DEPTH][-1]["bids"])} at {get_date_str(_data_record.data_record_list[DS_DEPTH][-1]["time"])}')
                        pass
                    _record, _status = _data_pack.get_data(None, end_ts)
                    _data_record.merge_data(_record)
                    print(
                        f'\tdata record 3: {len(_data_record.data_record_list[DS_DEPTH])}\t{len(_data_record.data_record_list[DS_DEPTH][-1]["bids"])} at {get_date_str(_data_record.data_record_list[DS_DEPTH][-1]["time"])}')

        self.queue_index = p_index
        return _data_record, _status

    pass


# 一个DataGroup保存多个DataQueue，代表当前可能用到的所有市场数据
class DataGroup(object):
    configs: [DataConfig()] = None
    data_group: [DataQueue()] = None

    def __init__(self, configs=None):
        self.configs = [DataConfig()] * 0
        self.data_group = [DataQueue()] * 0
        if configs is None:
            configs = [DataConfig()] * 0
        if configs is not None and len(configs) > 0:
            for _config in configs:
                self.add_config(_config)

    def add_config(self, config: DataConfig = None):
        if config is None:
            return
        for _config in self.configs:
            if _config.identify(config):
                return

        _new_config = DataConfig()
        _new_config.set_config(config=config.get_config())
        self.configs.append(_new_config)
        self.data_group.append(DataQueue(_new_config))

    def get_data(self, start_ts: int, end_ts: int):
        _data_records = []

        for dq in self.data_group:
            _record_list, _status = dq.get_data(start_ts, end_ts)
            if _record_list is not None:
                _data_records.append(_record_list)
        return _data_records

    pass


if __name__ == "__main__":
    settings = DataConfig(prime_coin="ETH", quote_coin="USDT", chain="", exchange="binance",
                          trade_type="spot", date_hour="2023051000", depth_level=DEFAULT_LEVEL,
                          source=S3_FILE, data_root="", file_type=LOG_FILE, compressed=True, pst=PST_NORMAL)

    """
    data_item = DataPack(config=settings)
    print("raw depth:", len(data_item.data_set[DS_DEPTH]), data_item.data_set[DS_DEPTH][0])
    if data_item.data_set[DS_TICKERS] is not None:
        print("raw tickers:", len(data_item.data_set[DS_TICKERS]), data_item.data_set[DS_TICKERS][0])
    if data_item.data_set[DS_TRADE] is not None:
        print("raw trades:", len(data_item.data_set[DS_TRADE]), data_item.data_set[DS_TRADE][0])

    data_item.recover_depth(need_loading=False, config=settings, snapshot=None)
    print("recovery depth:", len(data_item.data_set[DS_DEPTH]), data_item.data_set[DS_DEPTH][0])
    # data_item.data_set[DS_MARKET_PRICE] = get_market_price_list(data_item.data_set[DS_DEPTH])
    # print("market price:", len(data_item.data_set[DS_MARKET_PRICE]), data_item.data_set[DS_MARKET_PRICE][0])
    """

    settings.end_date_hour = "2023051022"
    next_date_hour = get_next_date_hours(settings.date_hour)
    pre_date_hour = get_next_date_hours(settings.date_hour, -1)
    data_in_use = DataQueue(config=settings)
    # data_in_use.data_queue[0].recover_depth()
    print("new data queue:", data_in_use, data_in_use.config.get_config())

    """
    s_ts = 1683652698052
    cur_data, status = data_in_use.get_data(s_ts)
    print("[get data] at:", s_ts, get_date_str(s_ts), "\n")
    if cur_data is not None:
        print(cur_data.get_data(DS_DEPTH)[0], len(cur_data.get_data(DS_TICKERS)), len(cur_data.get_data(DS_TRADE)), cur_data.data_record_list.keys(),
              len(cur_data.data_record_list), cur_data.get_attrib(), status, len(data_in_use.data_queue), data_in_use.queue_index,
              "\n")
    else:
        print(cur_data, status, "\n")
    """

    """"""
    s_ts = 1683690698052
    e_ts = 1683699698052
    cur_data, status = data_in_use.get_data(1683690698052, 1683699698052)
    print("[get data] from:", s_ts, "\tto:", e_ts, get_date_str(s_ts), get_date_str(e_ts), "\n")
    if cur_data is not None:
        print(len(cur_data.get_data(DS_DEPTH)[-1]['bids']), len(cur_data.get_data(DS_TICKERS)), len(cur_data.get_data(DS_TRADE)), cur_data.data_record_list.keys(),
              len(cur_data.data_record_list), cur_data.get_attrib(), status, len(data_in_use.data_queue), data_in_use.queue_index,
              "\n")
    else:
        print(cur_data, status, "\n")
    """"""

    """
    data_group = DataGroup(configs=[settings])
    print("data group:", data_group.configs[0].get_config(), len(data_group.data_group))

    record_list = data_group.get_data(s_ts, e_ts)
    if record_list is not None:
        print("\n", len(record_list), record_list[0].get_data(DS_DEPTH)[-1], record_list[0].get_data(DS_MARKET_PRICE)[-1])

    """



