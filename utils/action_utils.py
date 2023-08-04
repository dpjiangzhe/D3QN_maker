# import logging

ACTION_DIM = 2
ACTION_EDGE = 5


def loc_action(act, action_table):
    for i, t_value in enumerate(action_table):
        # print(i, act, t_value, action_table)
        if t_value is None:
            if act is None:
                return i
        elif act is not None:
            if act == t_value:
                return i


def action_table_encoding(actions, action_table):
    # 行动列表actions里的每个数值对应一轮行动中的一个维度
    # 行动的各维度的行动范围，保存在action_table的对应位置
    _a_dim = len(actions)
    _action_code = 0
    for i, act in enumerate(actions):
        nbase = len(action_table[i])
        act_index = loc_action(act, action_table[i])
        _action_code = _action_code * nbase + act_index
        # print(act_index, _action_code)
    return _action_code


def action_table_decoding(act_code, action_table):
    # 行动列表actions里的每个数值对应一轮行动中的一个维度
    # 行动的各维度的行动范围，保存在action_table的对应位置
    _action = []
    remain = act_code
    while len(_action) < len(action_table):
        i = len(action_table) - len(_action) - 1
        nbase = len(action_table[i])
        act_index = remain % nbase
        act = action_table[i][act_index]
        _action = [act, *_action]
        # print(act_code, i, nbase, act_index, act, _action)
        if remain > act_index:
            remain = int(remain / nbase)
        else:
            remain = 0
    return _action


def action_list_encoding(actions, action_edge):
    # action为0~action_edge，为常规挂单行动，编码调整到0～action_edge+1
    # action为None，则属于不挂单行动，编码为最大值action_edge+1
    _action_code = 0
    nbase = action_edge + 2
    for act in actions:
        act_modi = nbase - 1
        if act is not None:
            act_modi = act
        _action_code = _action_code * nbase + act_modi
    return _action_code


def action_list_decoding(act_code=0, action_dim=ACTION_DIM, action_edge=ACTION_EDGE):
    _action = []
    nbase = action_edge + 2
    remain = act_code
    while len(_action) < action_dim:
        act = remain % nbase
        act_modi = None
        if act < nbase - 1:
            act_modi = act
        _action = [act_modi, *_action]
        if remain > act:
            remain = int(remain / nbase)
        else:
            remain = 0
    return _action


def log_info(logger=None, msg=""):
    if logger is None:
        print(msg)
    else:
        logger.info(msg)


def store_history(value_list=None, history_data=None):
    if value_list is None or history_data is None:
        return None

    value_list = list.append(value_list, history_data)
    return value_list


if __name__ == '__main__':
    """
    for a1 in range(0, ACTION_EDGE+2):
        for a2 in range(0, ACTION_EDGE+2):
            act1 = a1
            if act1 == ACTION_EDGE+1:
                act1 = None
            act2 = a2
            if act2 == ACTION_EDGE+1:
                act2 = None
            action = [act1, act2]
            action_code = action_list_encoding(action, ACTION_EDGE)
            action_list = action_list_decoding(action_code, ACTION_DIM, ACTION_EDGE)
            print("(", a1, ",", a2, "):", action, "==>", action_code, "==>", action_list)
    """

    a_table = [[-3, -2, -1, 0, None], [0, 1, 2, 3, None], [1.0, 3.0, 0.5]]
    for a1 in a_table[0]:
        for a2 in a_table[1]:
            for a3 in a_table[2]:
                action = [a1, a2, a3]
                action_code = action_table_encoding(action, a_table)
                action_list = action_table_decoding(action_code, a_table)
                print("(", a1, ",", a2, ",", a3, "):", action, "==>", action_code, "==>", action_list)
