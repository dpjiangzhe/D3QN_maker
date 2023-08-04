#encoding=utf-8

import time
import datetime
import pytz

DATETIME_FORMAT = "%Y-%m-%d %H:%M:%S.%fZ"
DATETIME_FORMAT1 = "%Y-%m-%d %H:%M"
DATETIME_FORMAT2 = "%Y-%m-%dT%H:%M"
DATETIME_FORMAT3 = "%Y/%m/%d %H:%M"
DATETIME_FORMAT4 = "%Y%m%d%H"

# 东8区时区 Asia/Chongqing Asia/Shanghai
TZ_8 = pytz.timezone('Etc/GMT-8')
# 国际时间时区
TZ_0 = pytz.utc

def dt2unix(t, mill_type=1):
    # 直接使用 t.timestamp()得到时间戳
    # datetime.datetime()中可以设置时区，参数是：tzinfo，例如：
    # datetime.datetime(2022,10,10,8,46,59, tzinfo=pytz.utc)
#     if mill_type==1000:
#         return time.mktime(t.timetuple()) * mill_type + t.microsecond/mill_type
#     else:
#         return time.mktime(t.timetuple()) * mill_type
    return int(t.timestamp() * mill_type)
    

def unix2dt(t, mill_type=1, tz=None):
    '''
        tz: 表示时区
    '''
    return datetime.datetime.fromtimestamp(t / mill_type, tz=tz)

def dt2str(dt, dt_form='%Y-%m-%dT%H_%M'):
    return dt.strftime(dt_form)

def str2unix(time_str,dt_form=DATETIME_FORMAT, mill_type=1000, tz=None):
    dt = str2dt(time_str,dt_form)
    if tz is not None:
        dt = dt.replace(tzinfo=tz)
    return dt2unix(dt, mill_type)

def str2dt(time_str, dt_form="%Y-%m-%d %H:%M"):
    return datetime.datetime.strptime(time_str, dt_form)

def truncate_minute(d):
    '''
    将时间按分钟截取
    '''
    if isinstance(d, datetime.datetime):
        return datetime.datetime(d.year, d.month, d.day, d.hour, d.minute)
    if isinstance(d, datetime.date):
        return datetime.datetime(d.year, d.month, d.day)
    raise ValueError("parament should be datetime or date")

def truncate_date(d):
    '''
    将时间按日期截取
    '''
    return datetime.datetime(d.year, d.month, d.day)

def truncate_hour(d):
    return datetime.datetime(d.year, d.month, d.day, d.hour)

def unix2seconds(t):
    '''
       将 t 时间戳，统一转化为单位为秒
    '''
    if len(str(int(t))) == 10:
        return t
    elif len(str(int(t))) == 13:
        return float(t) / 1000

import re
def time2unix(num):
    """
    输入datetime格式或者"%Y-%m-%d %H:%M:%S.%f"格式，输出unix时间戳格式，输出单位为毫秒
    """
    try:
        b = re.findall("\D", num)
        #print('str')
        t = datetime.datetime.strptime(num, "%Y-%m-%d %H:%M:%S.%f")
        dt = t.replace(tzinfo=TZ_8)
        return dt2unix(dt, mill_type=1000,)
    except:
        #print('dt')
        return dt2unix(num,mill_type=1000)


#时间戳转化为字符串
def timestamp2string(tp):
    """
    author : 王培
    date : 2022/07/20
    input : timestamp
    output : Y-M-D H-M-S.f (例：2022-02-20 09:04:22.125)
    """
    tp = tp/1000
    d = datetime.datetime.fromtimestamp(tp)
    dt = d.strftime("%Y-%m-%d %H:%M:%S.%f")
    return dt

#时间戳转化为字符串, 不需要年月日， 只要时分秒
def timestamp2string_hms(tp):
    """
    author : 张鑫
    date : 2022/11/22
    input : timestamp
    output : Y-M-D H-M-S.f (例：09:04:22)
    """
    tp = tp/1000
    d = datetime.datetime.fromtimestamp(tp)
    dt = d.strftime("%H:%M:%S")
    return dt

#时间转化为时间戳
def string2timestamp(dt):
    """
    author : 王培
    date : 2022/07/20
    input : Y-M-D H-M-S.f (例：2022-02-20 09:04:22.125)
    output : timestamp
    """
    t = datetime.datetime.strptime(dt, "%Y-%m-%d %H:%M:%S.%f")
    tp= int(time.mktime(t.timetuple()) * 1000.0 + t.microsecond / 1000.0)
    return tp

def pro_timezone(start_time,end_time):
    if str(start_time.tzinfo)[:7]=='Etc/GMT':
            start_time_ = unix2dt(dt2unix(start_time)+8*60*60)
            end_time_ = unix2dt(dt2unix(end_time)+8*60*60)
    elif str(end_time.tzinfo) in ['Asia/Shanghai','Asia/Harbin','Asia/Chongqing']:
        start_time_ = start_time
        end_time_ = end_time
        print('应用UTC+8时区')
    elif end_time.tzinfo==None:
        start_time_ = unix2dt(dt2unix(start_time)+8*60*60)
        end_time_ = unix2dt(dt2unix(end_time)+8*60*60)
        print('时间数据未输入时区，按照UTC时间处理')
    else:
        print('输入的时区有误，请重新输入，格式如：Etc/GMT-8')
        return 0,0
    return start_time_,end_time_

# 将dataframe中的指定列的时间戳转化为字符串, 如果需要其他时间格式，更改timestamp2string_hms即可
def timestamp_data_process(data, feature='e'):
    """
    author : 张鑫
    date : 2022/11/22
    input : 
            data: dataframe
            feature: 时间戳所在列，如"e", "t","tp"等，默认为"e"
    output : 
            返回data, 其中时间戳所在列，如"e"转换为需要的格式 (例：09:04:22)
    """
    for i in range(len(data)):
        data[feature].iloc[i] = timestamp2string_hms(data[feature].iloc[i])
    return data

#将df中指定时间序列改为str格式，便于画图
def dftp2date(df,timetype='tp',tz=None):
    dt_list = []
    for i in range(len(df)):
        dt_list.append(dt2str(unix2dt(df[timetype].values[i],1000,tz=tz),dt_form="%Y-%m-%d %H:%M:%S.%f"))
    df['showtime'] = dt_list
    return df

def get_datelist(start_time,end_time):
    datelist = []
    if start_time > end_time:
        print('您输入的时间有误，请重新输入')
    else:
        tp = dt2unix(start_time, 1000)
        endtp = dt2unix(end_time, 1000)
        while tp <= endtp:
            date = dt2str(unix2dt(tp,1000), DATETIME_FORMAT4)
            datelist.append(date)
            tp += 60*60*1000
    return datelist


if __name__ == '__main__':
    a = '2023010910'
    dt = str2dt(a, '%Y%m%d%H')
    print(dt)