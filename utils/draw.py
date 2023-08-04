import os.path
from datetime import datetime

import matplotlib.pyplot as plt
import pandas as pd

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def draw(df, lines, x_label, y_label, title='pic'):
    plt.figure(figsize=(10, 6))  # Adjust the figure size as per your preference
    for line in lines:
        plt.plot(df[line], label=line)

    # Add labels and title
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)

    # Add legend
#     plt.legend()

    # Display the plot
    plt.show()


def draw_acc_history(fp, date_str):
    # 读取acc_history 数据
    fn = fp + date_str + "/acc_history.csv"
    if not os.path.exists(fn):
        return
    acc_history = pd.read_csv(fn)

    # 读取训练中的info信息
    fn = fp + date_str + "/info.txt"
    with open(fn, 'r') as f:
        contents = f.read()
        print(contents)

    # 对数据进行必要的转换
    acc_history['time'] = pd.to_datetime(acc_history['end_ts'], unit='ms')
    # 手续费
    acc_history['fee'] = acc_history['tariff'] * acc_history['m_price']
    # 最大回撤 百分比
    acc_history['draw_down'] = acc_history['drawdown'] * 100
    # 每一周期的成交量
    amount = (acc_history['filled_buy'] + acc_history['filled_sell']).values / acc_history['base_inv'].values[0]

    # 计算累计成交量
    suma = []
    tmp = 0
    for i in range(len(amount)):
        tmp += amount[i]
        suma.append(tmp)

    acc_history['amount'] = suma

    # mdd : max draw down
    # 计算最大回撤点，然后在图中标出
    mdd_index = acc_history['draw_down'].idxmax()
    _index = mdd_index
    while _index >= 0:
        if acc_history['draw_down'].values[_index] < acc_history['draw_down'].values[mdd_index]:
            _index += 1
            break
        else:
            _index -= 1
        print(acc_history['draw_down'].values[_index], acc_history['draw_down'].values[mdd_index])
    print(_index, mdd_index)
    mdd_index = _index
    mdd_value = acc_history.loc[mdd_index]['draw_down']
    mdd_x = acc_history.loc[mdd_index]['time']
    mdd_x = pd.to_datetime(mdd_x, unit="ms").strftime("%B %d %Y %X.%f")
    mdd_y = acc_history.loc[mdd_index]['t_inc']

    # 成交量，base_inv 单独一个比例尺
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    ts_list = [ts.strftime("%B %d %Y %X.%f") for ts in acc_history['time']]
    # ts_list = [i for i in acc_history.index]
    # print(ts_list[0])
    trace1 = go.Scatter(x=ts_list, y=acc_history['amount'], name="trading amount ratio")
    fig.add_trace(trace1, secondary_y=False)

    fig.add_trace(go.Scatter(x=ts_list, y=acc_history['base_inv'], name="base inventory"), secondary_y=False)
    origin_inv = acc_history['base_inv'].values[0]
    trace2 = go.Scatter(x=ts_list, y=[origin_inv]*len(ts_list), name="original base inv", yaxis="y3")
    fig.add_trace(trace2, secondary_y=False)
    fig.update_layout(yaxis3=dict(title="", anchor="free", overlaying="y", side="left", position=0.0, range=[origin_inv, origin_inv]))

    # t_inc放入另一个比例尺
    fig.add_trace(go.Scatter(x=ts_list, y=acc_history['t_inc'], name="trading income"), secondary_y=True)

    # 加入最大回撤点
    fig.add_trace(px.scatter(x=[mdd_x], y=[mdd_y], size=[100], text=[f'Max Draw Down:{round(mdd_value, 3)}%']).data[0],
                  secondary_y=True)

    anno = [dict(x=1, xref='paper', y=1, yref='paper', text=contents,
                 showarrow=False, ), ]
    fig.update_layout(
        title_text=f"{contents}"
        # annotations=anno
    )

    fig.update_xaxes(title_text="Time")
    fig.update_yaxes(title_text=f'<b>Amount(in base)</b>/ratio', secondary_y=False)
    fig.update_yaxes(title_text="<b>Profit(in quote)<b>", secondary_y=True)

    fig.show()


def draw_trading_history(fp, date_str):
    # 读取acc_history 数据
    fn = fp + date_str + "/acc_history.csv"
    if not os.path.exists(fn):
        return
    acc_history = pd.read_csv(fn)
    # 对数据进行必要的转换
    acc_history['time'] = pd.to_datetime(acc_history['end_ts'], unit='ms')

    fn = fp + date_str + "/tra_history.csv"
    if not os.path.exists(fn):
        return
    tra_history = pd.read_csv(fn)

    # 成交量，base_inv 单独一个比例尺
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    ts_list = [ts.strftime("%B %d %Y %X.%f") for ts in acc_history['time']]
    trace1 = go.Scatter(x=ts_list, y=acc_history['m_price'], name="market price")
    fig.add_trace(trace1, secondary_y=True)
    trace2 = go.Scatter(x=ts_list, y=acc_history['t_inc'], name="trading income")
    fig.add_trace(trace2, secondary_y=False)
    trace3 = go.Scatter(x=ts_list, y=tra_history['price_buy'], name="price buy")
    fig.add_trace(trace3, secondary_y=True)
    trace4 = go.Scatter(x=ts_list, y=tra_history['price_sell'], name="price sell")
    fig.add_trace(trace4, secondary_y=True)

    fig.update_xaxes(title_text="Time")
    fig.update_yaxes(title_text=f'<b>price</b>', secondary_y=True)
    fig.update_yaxes(title_text="<b>profit(in quote)<b>", secondary_y=False)

    fig.show()
    pass


if __name__ == "__main__":
    # df = pd.read_csv("../data/history/202306071550/t_acc_history.csv")
    # df.dropna(axis=0, how="any", inplace=True)
    # draw(df, ['loss'], 'time', 'value')

    draw_acc_history("../data/history/", "202307020332_batched")

    # draw_trading_history("../data/history/", "202306131850_batched")


