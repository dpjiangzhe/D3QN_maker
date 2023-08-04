from datetime import datetime
import argparse
import json
import os
import copy

import numpy as np
import pandas as pd
import time
import torch
# import gymnasium as gym

import market_env as me
# import DDPG
import D3QN
import utils.datatools as dts
from utils import action_utils as ats
import utils.filetools as fts

# import proto.py.sdk.logger as lg

MAX_EPISODES = 10
MAX_EP_STEPS = 600
LR_A = 1e-4  # learning rate for actor
LR_C = 1e-4  # learning rate for critic
GAMMA = 0.9  # reward discount
REPLACE_ITER_A = 800
REPLACE_ITER_C = 700
MEMORY_CAPACITY = 1000
LOAD = False

STATE_DIM = 29
ACTION_DIM = 2
ACTION_EDGE = 5
ACTION_TYPE = int(pow((ACTION_EDGE + 1) * 2, ACTION_DIM))
ACTION_BOUND = [-ACTION_EDGE, ACTION_EDGE]

INTERVAL = 1000  # 窗口宽度，1000毫秒=1秒
SECOND_NS = 1000 * 1000 * 1000  # 1秒包含的纳秒数

RM_TRAIN_AND_BACKTEST = 0
RM_TRAIN = 1
RM_BACKTEST = 2

seed = 1
torch.manual_seed(seed)
np.random.seed(seed)
torch.set_default_dtype(torch.float)


# 根据状态state，让model产生行动
def generate_actions(_settings, model, state, is_train=True, action_table=None):
    if not model.need_coding:
        # model直接产生行动表
        action = model.choose_action(state, is_train)
        if action_table is not None:
            action_code = ats.action_table_encoding(action, action_table)
        else:
            action_code = ats.action_list_encoding(action, _settings["action_edge"])
    else:
        # model产生一个行动编码，译码转换成行动表
        action_code = model.choose_action(state, is_train)
        if action_table is not None:
            action = ats.action_table_decoding(action_code, action_table)
        else:
            action = ats.action_list_decoding(action_code, _settings["action_dim"], _settings["action_edge"])
            # 买入价不高于盘口价，所以均为负值
            if action[0] is not None:
                action[0] = 0 - action[0]

    return action, action_code


def train(tenv: me.MarketEnv, model, file_path: str,
          training_steps=0, training_actions=1, span=1, show=False):
    var = 8.  # control exploration
    bound = [-tenv.settings["action_edge"], tenv.settings["action_edge"]]
    # bound = [-1, 1]
    # stage = "prepare"
    stage = "learning"
    tenv.reset(show=show, training=True, data_reset=False)

    a_data = tenv.get_account_data()
    action_states = [copy.copy(a_data)] * tenv.settings["action_type"]
    _report_msg = ""

#    for ep in range(epoch):
#    s = tenv.reset(show=show, training=True, clear_history=False)

    # INTERVAL为1000，表示1秒
    # interval
    rolling_wnd = span
    interval = INTERVAL * rolling_wnd
    training_end_index = tenv.market_df[0].shape[0]
    learning_loss = None
    # if training_steps > 0:
    #    training_end_index = training_steps * rolling_wnd

    for _index in range(0, training_end_index, rolling_wnd):
        _m_index = _index
        # if training_steps > 0:
        # 随机挑选执行数据
        #    m_index = np.random.randint(0, tenv.market_df.shape[0])
        #    tenv.market_index = m_index
        t, ts = tenv.get_market_state(_m_index)
        tenv.clock = ts

        # 产生交易动作
        end_ts = ts + interval

        action_list = []
        action_code_list = []

        if training_steps <= 0 or training_actions <= 1:
            a0 = [0, 0, 0]
            a = [0, 0, 0]
            a_code = 0
            if stage == "learning":
                a0, a_code = generate_actions(tenv.settings, model, t, action_table=tenv.settings["action_table"])
            if not model.need_coding:
                a1 = np.clip(np.random.normal([a0[0], a0[1]], var), bound[0], bound[1])  # 在动作选择上添加随机噪声
                if tenv.settings["action_dim"] == 3:
                    a1.append(a0[2])
                a0 = a1
                a_code = ats.action_table_encoding(a0)
            if tenv.settings["action_dim"] == 3:
                a = a0
            else:
                a[0], a[1], a[2] = a0[0], a0[1], 1.0
            action_list.append(a)
            action_code_list.append(a_code)
        else:
            # 随机挑选动作
            for _i in range(training_actions):
                env_set = tenv.settings
                if model.need_coding:
                    a_code = np.random.randint(0, env_set["action_type"])

                    if tenv.settings["action_table"] is None:
                        a = ats.action_list_decoding(a_code, tenv.action_dim, tenv.action_edge)
                        if a[0] is not None:
                            a[0] = 0 - a[0]
                    else:
                        a = ats.action_table_decoding(a_code, tenv.settings["action_table"])
                else:
                    # 随机产生每一维动作
                    if tenv.settings["action_table"] is None:
                        a = np.random.randint(-model.action_edge, model.action_edge, model.action_dim)
                    else:
                        a = []
                        for _a_t in tenv.settings["action_table"]:
                            act_index = np.random.randint(0, len(a_t) - 1)
                            act = _a_t[act_index]
                            a.append(act)
                    a_code = ats.action_list_encoding(a, model.action_edge)
                if tenv.settings["action_dim"] == 2:
                    a.append(1.0)
                action_list.append(a)
                action_code_list.append(a_code)
            pass

        # 保存执行动作前的现场
        c_a_data = tenv.get_account_data()
        for _act_index, _act in enumerate(action_list):
            # 恢复动作前的现场
            a_code = action_code_list[_act_index]
            if training_steps > 0 and training_actions > 1:
                _a_index = np.random.randint(0, tenv.settings["action_type"])
                a_data = action_states[_a_index]
                if a_data is not None and a_data.market_index[0] > 0:
                    a_data.market_index[0] = _m_index
                    tenv.set_account_data(a_data)
                else:
                    tenv.set_account_data(c_a_data)

            s, r, a, end_ts, s_ = tenv.step(_act[0], _act[1], _act[2], ts, end_ts)
            if model.need_coding:
                model.remember(s, action_code_list[_act_index], r, s_, False)
            else:
                model.remember(s, a, r, s_, False)

            # if model.pointer <= MEMORY_CAPACITY:
            #    print("memory size:", model.pointer)
            if model.pointer > MEMORY_CAPACITY:
                # learn
                var *= .9995  # decay the action randomness
                stage = "learning"
                tenv.learning_loss = float(model.learn().data)
                pass
            print("training:", _index, "/", tenv.market_df[0].shape[0], "\tloss:", tenv.learning_loss,
                  "\tinfo:", r, a, end_ts, tenv.account.base_inv, tenv.account.quote_inv, tenv.profit,
                  tenv.max_drawdown)

            # 保存动作后的行动状态
            if training_steps > 0:
                action_states[_act_index] = copy.copy(tenv.get_account_data())
        pass

    m_price, _, _ = tenv.get_market_price()
    o_price, _, _ = tenv.get_market_price(0)
    _msg = f'\ntrain statics: {tenv.order_filled_buy}/{tenv.order_counts_buy}\t{tenv.order_filled_sell}/{tenv.order_counts_sell}\tsuccess pairs: {tenv.pair_success}; amount: {tenv.pair_success_amount}'
    result_msg = _msg
    ats.log_info(logger, _msg)
    _msg = f'\ntrain statics: {tenv.filled_buy_amount}\t{tenv.filled_sell_amount}\t{(tenv.order_counts_buy + env.order_counts_sell) * tenv.account.base_portion}'
    result_msg += _msg
    ats.log_info(logger, _msg)
    net_inc = (
                          tenv.inventory - tenv.account_origin.base_inv) * m_price + tenv.account.quote_inv - tenv.account_origin.quote_inv
    profit = net_inc / tenv.account_origin.value
    _msg = f'\ntrain statics: {tenv.inventory}/{tenv.account_origin.base_inv}\t{tenv.account.quote_inv}/{tenv.account_origin.quote_inv}' \
           f'\t{tenv.account.base_c}/{tenv.account.quote_c}={m_price}' \
           f'\n\tnet_inc: {net_inc} = profit_inv: {tenv.profit_inv} + profit_trade: {tenv.profit_trade} {tenv.account.quote_c}' \
           f'\n\ttrading profit: {profit * 100}%\ttotal income: {tenv.total_profit}\ttotal ar: {(tenv.total_ar/tenv.account_origin.value) * 100}%' \
           f'\ttrading drawdown: {tenv.max_drawdown * 100}%' \
           f'\ttariff: {tenv.tariff}{tenv.account.base_c}\ttariff(quote): {tenv.tariff_quote}{tenv.account.quote_c}' \
           f'\nturnover rate: {tenv.turnover_rate}\tavg. tr: {tenv.turnover_rate_per_day}/day\tmode eval: {tenv.evaluation}'
    result_msg += _msg
    _msg = f'\ntime consuming:\tloc:{tenv.timer["loc"] / SECOND_NS}\tget price:{tenv.timer["get_p"] / SECOND_NS}' \
           f'\tprocessing:{tenv.timer["other"] / SECOND_NS}\tstore:{tenv.timer["store_h"] / SECOND_NS}\n'
    result_msg += _msg
    ats.log_info(logger, _msg)
    log_fn = "result.log"

    if not settings["save2s3"]:
        with open(file_path + log_fn, 'a') as log_file:
            log_file.write(result_msg)
    _report_msg += result_msg + "\r\n"

    # break

    """
    keys = env.market_df.columns.values.tolist()
    model.save_models(mod_path, ep, keys)
    for var_name in model.q_eval.state_dict():
        print(var_name, "\t", model.q_eval.state_dict()[var_name])
    for var_name in model.q_target.state_dict():
        print(var_name, "\t", model.q_target.state_dict()[var_name])
    """
    return model, report_msg


def backtest(tenv: me.MarketEnv, model, file_path, span=1, show=False, clear_history=True, data_reset=False,
             acc_reset=False, random_test=False):
    tenv.reset(show=show, training=False, clear_history=clear_history, data_reset=data_reset, acc_reset=acc_reset)
    rolling_wnd = span
    interval = INTERVAL * rolling_wnd

    # test for random
    _seed = abs(int(time.time_ns() / 1000000000))
    torch.manual_seed(_seed)
    np.random.seed(_seed)

    for _index in range(tenv.market_index[0], tenv.market_df[0].shape[0], rolling_wnd):
        t, ts = tenv.get_market_state(_index)
        tenv.clock = ts

        # 产生交易动作
        if not random_test:
            a, a_code = generate_actions(tenv.settings, model, t, False, tenv.settings["action_table"])
        else:
            a_code = np.random.choice(model.action_space)
            a = ats.action_table_decoding(a_code, tenv.settings["action_table"])
        # test: 固定盘口挂单
        # a = [0, 0, 1.0]

        if len(a) == 2 or tenv.settings["action_dim"] == 2:
            a.append(1.0)
        end_ts = ts + interval
        s, r, a, end_ts, s_ = tenv.step(a[0], a[1], a[2], ts, end_ts)
        profit, price = tenv.calc_profit()
        print("simulating: (", _index, "--", tenv.market_index[0], ") /", tenv.market_df[0].shape[0], r, a, ts, profit, price,
              tenv.account.base_inv, tenv.account.quote_inv, tenv.profit, tenv.max_drawdown)
        s = s_

    m_price, _, _ = tenv.get_market_price()
    o_price, _, _ = tenv.get_market_price(0)
    _msg = f'\nbacktest statics: {tenv.order_filled_buy}/{tenv.order_counts_buy}\t{tenv.order_filled_sell}/{tenv.order_counts_sell}\tsuccess pairs: {tenv.pair_success}; amount: {tenv.pair_success_amount}'
    result_msg = _msg
    ats.log_info(logger, _msg)
    _msg = f'\nbacktest statics: {tenv.filled_buy_amount}\t{tenv.filled_sell_amount}\t{(tenv.order_counts_buy + env.order_counts_sell) * tenv.account.base_portion}'
    result_msg += _msg
    ats.log_info(logger, _msg)
    net_inc = (
                          tenv.inventory - tenv.account_origin.base_inv) * m_price + tenv.account.quote_inv - tenv.account_origin.quote_inv
    profit = net_inc / tenv.account_origin.value
    _msg = f'\nbacktest statics: {tenv.inventory}/{tenv.account_origin.base_inv}\t{tenv.account.quote_inv}/{tenv.account_origin.quote_inv}' \
           f'\t{tenv.account.base_c}/{tenv.account.quote_c}={m_price}' \
           f'\n\tnet_inc: {net_inc} = profit_inv: {tenv.profit_inv} + profit_trade: {tenv.profit_trade} {tenv.account.quote_c}' \
           f'\n\ttrading profit: {profit * 100}%\ttotal income: {tenv.total_profit}\ttotal ar: {(tenv.total_ar/tenv.account_origin.value) * 100}%' \
           f'\ttrading drawdown: {tenv.max_drawdown * 100}%' \
           f'\ttariff: {tenv.tariff}{tenv.account.base_c}\ttariff(quote): {tenv.tariff_quote}{tenv.account.quote_c}' \
           f'\nturnover rate: {tenv.turnover_rate}\tavg. tr: {tenv.turnover_rate_per_day}/day\tmode eval: {tenv.evaluation}'
    result_msg += _msg
    _msg = f'\ntime consuming:\tloc:{tenv.timer["loc"] / SECOND_NS}\tget price:{tenv.timer["get_p"] / SECOND_NS}' \
           f'\tprocessing:{tenv.timer["other"] / SECOND_NS}\tstore:{tenv.timer["store_h"] / SECOND_NS}\n'
    result_msg += _msg
    ats.log_info(logger, _msg)
    log_fn = "result.log"
    if not settings["save2s3"]:
        with open(file_path + log_fn, 'a') as log_file:
            log_file.write(result_msg)
    print("[Backtest ended]", result_msg)
    return result_msg


if __name__ == '__main__':
    # os.environ['KMP_DUPLICATE_LIB_OK'] = True

    parser = argparse.ArgumentParser()

    REPLACEMENT = [
        dict(name='soft', tau=0.005),
        dict(name='hard', rep_iter=600)
    ][0]  # you can try different target replacement strategies

    logger = None
    # logger = lg.NewLogger("./logs/", "train.log", "24h")

    with open('conf/account.json') as f:
        acc_file = json.load(f)
    acc_info = me.AccountInfo(acc_file)

    with open('conf/system.json') as f:
        system_config = json.load(f)
    settings = system_config.get('environment')
    model_settings = system_config.get('model')
    tariff_rate = settings.get('tariff_rate')
    env = me.MarketEnv(settings)

    online = False
    batched = True
    running_mode = RM_BACKTEST
    if "running_mode" in settings:
        running_mode = settings["running_mode"]
    if "online" in settings:
        online = settings["online"]
    if "batched" in settings:
        batched = settings["batched"]
    models = settings["models"]
    start_date = settings["start_date"]
    end_date = settings["end_date"]

    filepath = "./data/"
    historyPath = filepath + "history/"
    model_path = "./models/"
    file_type = ".csv"
    if online:
        filepath = "market_maker/data/one_year/binance/"
        file_type = ".csv.gz"
    elif batched:
        filepath += "one_year/binance/"

    if settings["save2s3"]:
        parser.add_argument("--hosts", type=list, default=json.loads(os.environ["SM_HOSTS"]))
        parser.add_argument("--current-host", type=str, default=os.environ["SM_CURRENT_HOST"])
        parser.add_argument("--model-dir", type=str, default=os.environ["SM_MODEL_DIR"])
        parser.add_argument("--data-dir", type=str, default=os.environ["SM_CHANNEL_TRAINING"])
        parser.add_argument("--num-gpus", type=int, default=os.environ["SM_NUM_GPUS"])

        model_path = parser.parse_args().model_dir + "/"
        historyPath = parser.parse_args().data_dir + "/"
        """
        running_mode = RM_TRAIN_AND_BACKTEST
        fn = model_path + "test.txt"
        print("write file test...", fn)
        with open(fn, "a") as f:
            f.write(__model_path)
        print("read file test...", fn)
        with open(fn, "r") as f:
            _content = f.read()
        print("reading: ", _content)
        """

    device = D3QN.set_device(online)

    if running_mode == RM_TRAIN:
        exch = "_binance_"
        trade_pair = "eth_usdt"
        # date = "_2023040800"
        # acc_info = eth_acc

        # date_list = ["_2023051305", "_2023051517", "_2023051607"]
        date_list = ["_2023030310", "_2023040800", "_2023041308", "_2023041900", "_2023042107", "_2023050300",
                     "_2023051305", "_2023051517", "_2023051607"]
        # date_list = ["_2022080400"]

        if online or batched:
            # 指定日期范围(起止年月日)，产生date_list
            date_list = dts.generate_date_list(start_date, end_date)
            pass

        a_edge = settings["action_edge"]
        a_bound = [0, a_edge]
        if settings["action_table"] is None:
            a_type = int((a_edge + 2) ** settings["action_dim"])
        else:
            a_type = 1
            for a_t in settings["action_table"]:
                a_type *= len(a_t)
        settings["action_type"] = a_type
        start_ts = time.time_ns()
        date_time = datetime.now()
        date_str = date_time.strftime("%Y%m%d%H%M")

        if not settings["save2s3"] and not os.path.exists(historyPath):
            os.mkdir(historyPath)
        history_path = historyPath + date_str
        if batched:
            history_path += "_batched/"
        else:
            history_path += "/"
        if not settings["save2s3"] and not os.path.exists(history_path):
            os.mkdir(history_path)

        model_save_path = model_path + date_str + "/"
        if not settings["save2s3"] and not os.path.exists(model_save_path):
            os.mkdir(model_save_path)

        d3qn = None

        trading_time = 0
        report_msg = ""
        model_in_use = None
        network_dims1 = network_dims2 = []
        batch_size = 256
        for index, date in enumerate(date_list):
            env.reset(training=True)

            _date = date
            if online or batched:
                _date = date + "00"  # 目前线上数据：目录为日期，文件名为日期+00
            _filepath = filepath + trade_pair + date + "/"
            _filepath1 = filepath + trade_pair + "_uswap" + date + "/"
            file_tail = exch + trade_pair + _date + file_type
            file_tail1 = exch + trade_pair + "_uswap" + _date + file_type

            fn = _filepath + "features" + file_tail
            print("loading market data (features) ...", fn)
            if not env.load_market_data(fn, settings["keys"], online):
                # 无法获取数据
                print("failed...")
                continue
            fn = _filepath1 + "features" + file_tail1
            # print("loading market data (features) ...", fn)
            # if not env.load_market_data(fn, settings["keys"], online):
                # 无法获取数据
            #    print("failed...")
            #    continue

            env.prepare_data()
            fn = _filepath + "trade" + file_tail
            print("loading trade data ...", fn)
            if not env.load_trade_data(fn, online):
                # 无法获取数据
                print("failed...")
                continue

            fn = _filepath + "tickers" + file_tail
            print("loading ticker data ...", fn)
            if not env.load_ticker_data(fn, online):
                # 无法获取数据
                print("failed...")
                continue

            # fn = _filepath + "depth" + file_tail
            # print("loading depth data ...", fn)
            # if not env.load_depth_data(fn, online):
            # 无法获取数据
            # print("failed...")
            # continue

            fn = _filepath + "time_index" + file_tail
            print("loading index data ...", fn)
            env.load_index(fn, online)
            trading_time += env.market_df[0]["time"].values[-1] - env.market_df[0]["time"].values[0]

            if index == 0:
                if len(settings["keys"]) == 0:
                    s_dim = env.market_df[0].shape[1]
                else:
                    s_dim = len(settings["keys"])
                s_dim = (s_dim - 1) * (len(env.market_df)) + 1
                settings["state_dim"] = s_dim
                model_settings["state_dim"] = s_dim

                env.set_account_info(acc_info)
                print("acc info:", env.account)

                env.state_dim = settings["state_dim"]

                network_dims1 = model_settings["eval_dims"]
                network_dims2 = model_settings["target_dims"]
                batch_size = model_settings["batch_size"]
                # d3qn = D3QN.D3QN(alpha=0.0003, state_dim=settings["state_dim"], action_dim=a_type, action_edge=a_edge,
                #                 eval_dims=network_dims1, target_dims=network_dims2, ckpt_dir=model_save_path,
                #                 gamma=0.9, tau=0.005, epsilon=1.0,
                #                 eps_end=0.05, eps_dec=5e-4, max_size=10000, batch_size=batch_size)
                d3qn = D3QN.D3QN(model_settings, model_save_path)

                model_in_use = d3qn
                settings["model"] = model_in_use.name
                json_str = json.dumps(system_config)

                if not settings["save2s3"]:
                    with open(history_path + 'settings.json', 'w') as json_file:
                        json_file.write(json_str)
                    with open(model_save_path + 'settings.json', 'w') as json_file:
                        json_file.write(json_str)
                else:
                    # fts.save_object_to_s3(history_path + 'settings.json', json_str)
                    # fts.save_object_to_s3(model_save_path + 'settings.json', json_str)
                    # with open(historyPath + 'settings.json', 'w') as json_file:
                    #    json_file.write(json_str)
                    with open(model_path + 'settings.json', 'w') as json_file:
                        json_file.write(json_str)

                env.change_settings(settings)
                env.init_time = env.market_df[0]["time"].values[0]

                ats.log_info(logger, f'\n[SETTINGS] {json_str}')
                ats.log_info(logger, "[TRAIN]")

            epochs = settings["epochs"]
            time_span = settings["span"]

            random_acts = 1  # - int(env.data_counter / 10000)
            if random_acts <= 0:
                random_acts = 1
            model_in_use, _log_msg = train(env, model_in_use, history_path, 10000, random_acts, time_span)
            report_msg += _log_msg

            # 注意：考虑save2s3的情况
            env.save_history(history_path, model_save_path)

        # keys = env.market_df.columns.values.tolist()
        if not settings["save2s3"]:
            model_in_use.save_models(model_save_path, 0)
        else:
            model_in_use.save_models(model_path, 0)
            # _history_path = parser.parse_args().model_dir
            # fts.save_object_to_s3(history_path + "result.log", report_msg)
            with open(historyPath + 'result.log', 'w') as json_file:
                json_file.write(report_msg)

            print("saving result:", historyPath+'result.log')

        # for var_name in model_in_use.q_eval.state_dict():
        #    print(var_name, "\t", model_in_use.q_eval.state_dict()[var_name])
        # for var_name in model_in_use.q_target.state_dict():
        #    print(var_name, "\t", model_in_use.q_target.state_dict()[var_name])

        msg = f'model no.:{date_str}\tepoch:{settings["epochs"] - 1}\tfc_dims:{network_dims1}, {network_dims2}\tbatch_size:{batch_size}\n'
        msg += f'data counter: {env.data_counter}\tdata source: {exch + trade_pair}: {date_list}\n'
        msg += f'features: {settings["keys"]}\n'
        trading_hours = trading_time / 1000.0 / 60.0 / 60.0
        trading_days = int(trading_hours / 24)
        trading_hours -= trading_days * 24
        msg += f'Trading duration: {trading_days} days {trading_hours} hours\n'
        tr_ts = time.time_ns()
        msg += f'Time consuming: training: {(tr_ts - start_ts) / 1000.0 / 1000.0 / 1000.0 / 3600.0} hours\n'
        if settings["save2s3"]:
            # fts.save_object_to_s3(history_path + "info.txt", msg)
            # fts.save_object_to_s3(model_save_path + "info.txt", msg)
            with open(model_path + "info.txt", "w") as tag_file:
                tag_file.write(msg)
        else:
            with open(history_path + "info.txt", "w") as tag_file:
                tag_file.write(msg)
            with open(model_save_path + "info.txt", "w") as tag_file:
                tag_file.write(msg)

        ats.log_info(logger, msg)
    elif running_mode == RM_BACKTEST:
        model_path = "./models/"
        exch = "_binance_"
        trade_pair = "eth_usdt"
        # date = "_2023042107"

        # acc_info = eth_acc
        env.set_account_info(acc_info)
        print("acc info:", env.account)

        # models = ["202306130415"]
        date_list = ["_2023030310", "_2023040800", "_2023041308", "_2023041900", "_2023042107", "_2023050300",
                     "_2023051305", "_2023051517", "_2023051607"]
        # date_list = ["_2023041308"]

        if online or batched:
            # 指定日期范围(起止年月日)，产生date_list
            date_list = dts.generate_date_list(start_date, end_date)
            pass

        for m_index, m_no in enumerate(models):
            env.reset(show=True, training=False, clear_history=True, acc_reset=True)

            # 读入settings.json
            model_save_path = model_path + m_no + "/"
            json_str = ""
            if os.path.exists(model_save_path + 'settings.json'):
                with open(model_save_path + 'settings.json', 'r') as json_file:
                    json_str = json_file.read()
            if json_str == "":
                continue

            print("backtest for model: ", m_no, m_index)
            # 读入特征键表
            system_config = json.loads(json_str)
            settings = system_config.get("environment")
            model_settings = system_config.get("model")
            json_str = ""
            feature_keys = []
            # if os.path.exists(model_save_path + 'feature_keys.json'):
            #    with open(model_save_path + 'feature_keys.json', 'r') as json_file:
            #        json_str = json_file.read()
            # if json_str != "":
            #     feature_keys = json.loads(json_str)
            s_dim = settings["state_dim"]
            if "keys" in settings:
                feature_keys = settings["keys"]

            start_ts = time.time_ns()
            date_time = datetime.now()
            date_str = date_time.strftime("%Y%m%d%H%M")

            if "save2s3" not in settings:
                settings["save2s3"] = False
            if not settings["save2s3"] and not os.path.exists(historyPath):
                os.mkdir(historyPath)
            history_path = historyPath + date_str
            if batched:
                history_path += "_batched/"
            else:
                history_path += "/"
            if not settings["save2s3"] and not os.path.exists(history_path):
                os.mkdir(history_path)

            a_type = settings["action_type"]
            a_edge = settings["action_edge"]
            epochs = settings["epochs"]
            time_span = settings["span"]

            network_dims1 = model_settings["eval_dims"]
            network_dims2 = model_settings["target_dims"]
            batch_size = model_settings["batch_size"]
            # d3qn = D3QN.D3QN(alpha=0.0003, state_dim=settings["state_dim"], action_dim=a_type, action_edge=a_edge,
            #                 eval_dims=network_dims1, target_dims=network_dims2, ckpt_dir=model_save_path, gamma=0.9,
            #                 tau=0.005, epsilon=1.0,
            #                 eps_end=0.05, eps_dec=5e-4, max_size=10000, batch_size=batch_size)
            d3qn = D3QN.D3QN(model_settings, model_save_path)

            model_in_use = d3qn

            model_epoch = settings["epochs"] - 1
            if model_epoch < 0:
                model_epoch = 0
            model_in_use.load_models(model_save_path, model_epoch)
            env.load_scalers(model_save_path, "scalers.pkl")

            settings["model"] = model_in_use.name
            settings["tariff_rate"] = tariff_rate
            if "action_table" not in settings:
                settings["action_table"] = None

            settings["start_date"] = start_date
            settings["end_date"] = end_date

            env.change_settings(settings)
            json_str = json.dumps({"environment": settings, "model": model_settings})
            with open(history_path + 'settings.json', 'w') as json_file:
                json_file.write(json_str)

            ats.log_info(logger, f'\n[SETTINGS] {json_str}')
            ats.log_info(logger, "[BACKTEST]")

            trading_time = 0
            for _d_index, date in enumerate(date_list):
                _date = date
                if _date == "":
                    continue
                if online or batched:
                    _date = date + "00"  # 目前线上数据：目录为日期，文件名为日期+00
                _filepath = filepath + trade_pair + date + "/"
                _filepath1 = filepath + trade_pair + "_uswap" + date + "/"
                file_tail = exch + trade_pair + _date + file_type
                file_tail1 = exch + trade_pair + "_uswap" + _date + file_type

                env.reset(show=True, training=False)

                fn = _filepath + "features" + file_tail
                print("loading market data (features) ...", fn)
                if not env.load_market_data(fn, settings["keys"], online):
                    # 无法获取数据
                    print("failed...")
                    continue
                fn = _filepath1 + "features" + file_tail1
                # print("loading market data (features) ...", fn)
                # if not env.load_market_data(fn, settings["keys"], online):
                    # 无法获取数据
                #    print("failed...")
                #    continue

                env.prepare_data()
                fn = _filepath + "trade" + file_tail
                print("loading trade data ...", fn)
                if not env.load_trade_data(fn, online):
                    # 无法获取数据
                    print("failed...")
                    continue

                fn = _filepath + "tickers" + file_tail
                print("loading ticker data ...", fn)
                if not env.load_ticker_data(fn, online):
                    # 无法获取数据
                    print("failed...")
                    continue

                # fn = _filepath + "depth" + file_tail
                # print("loading depth data ...", fn)
                # if not env.load_depth_data(fn):
                # 无法获取数据
                # print("failed...")
                # continue

                fn = _filepath + "time_index" + file_tail
                print("loading index data ...", fn)
                env.load_index(fn, online)

                trading_time += env.market_df[0]["time"].values[-1] - env.market_df[0]["time"].values[0]

                random_test = False
                if _d_index == 0:
                    env.init_time = env.market_df[0]["time"].values[0]
                    backtest(env, model_in_use, history_path, time_span, show=True, clear_history=True, data_reset=False,
                             acc_reset=True, random_test=random_test)
                else:
                    backtest(env, model_in_use, history_path, time_span, show=True, clear_history=False,
                             data_reset=False, acc_reset=False, random_test=random_test)

                # 注意：考虑save2s3的情况
                env.save_history(history_path)
                env.reset(show=True, training=False)

            msg = f'model no.:{m_no}\tepoch:{settings["epochs"] - 1}\tfc_dims:{network_dims1}, {network_dims2}\tbatch_size:{batch_size}\r\n'
            msg += f'data counter: {env.data_counter}\tdata source: {exch + trade_pair}: {date_list}\r\n'
            msg += f'features: {feature_keys}\r\n'
            trading_hours = trading_time / 1000.0 / 60.0 / 60.0
            trading_days = int(trading_hours / 24)
            trading_hours -= trading_days * 24
            msg += f'Trading duration: {trading_days} days {trading_hours} hours\n'
            bt_ts = time.time_ns()
            msg += f'Time consuming: backtesting: {(bt_ts - start_ts) / 1000.0 / 1000.0 / 1000.0 / 3600.0} hours\r\n'
            if settings["save2s3"]:
                fts.save_object_to_s3(history_path + "info.txt", msg)
                # fts.save_object_to_s3(model_save_path + "info.txt", msg)
            else:
                with open(history_path + "info.txt", "w") as tag_file:
                    tag_file.write(msg)
                # with open(model_save_path + "info.txt", "w") as tag_file:
                #    tag_file.write(msg)

            ats.log_info(logger, msg)

            time.sleep(30)

            """
            for var_name in model_in_use.q_eval.state_dict():
                print(var_name, "\t", model_in_use.q_eval.state_dict()[var_name])
            for var_name in model_in_use.q_target.state_dict():
                print(var_name, "\t", model_in_use.q_target.state_dict()[var_name])
            """

        pass
