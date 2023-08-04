import numpy as np

from train import generate_actions
from train import MEMORY_CAPACITY

MODEL_UPD_PERIOD = 24 * 60 * 60 * 1000 * 1000


# 定期调用training_online，调用时准备好一个新的模型备用于重新训练；输入参数包括上一次行动的相关数据，以及当前的状态
def training_online(model, n_model, o_state, n_state, action, r, var, ts_now, ts_upd_model, model_path, settings, keys):
    # 检查是否需要更新model
    if ts_now - ts_upd_model >= MODEL_UPD_PERIOD and n_model is not None:
        # 保存模型更新数据
        model.save_models(model_path, 0, keys)
        # 重置模型
        model = n_model
        ts_upd_model = ts_now

    # 保存状态
    model.remember(n_state, action, r, o_state, False)
    # 学习
    if model.pointer > MEMORY_CAPACITY:
        var *= .9995  # decay the action randomness
        model.learn()

    # 产生新动作
    a, a_code = generate_actions(settings, model, n_state, False, settings["action_table"])
    if not model.need_coding:
        bound = [-settings["action_edge"], settings["action_edge"]]
        a = np.clip(np.random.normal(a, var), bound[0], bound[1])  # 在动作选择上添加随机噪声

    return a, a_code, ts_upd_model, var, model


# 滚动训练放在调用此函数的地方进行，每次行动前把上次行动的相关数据及当前状态存入模型，并进行学习
def generate_signal_online(model, state, ts_now, ts_upd_model, settings, model_path, model_epoch):
    # 检查是否需要更新model
    if ts_now - ts_upd_model >= MODEL_UPD_PERIOD:
        # 检查模型更新数据
        # 重新加载模型
        model.load_models(model_path, model_epoch)
        ts_upd_model = ts_now

    # 产生交易信号
    a, a_code = generate_actions(settings, model, state, False, settings["action_table"])

    # 返回交易信号
    return a, a_code, ts_upd_model
