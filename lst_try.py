import time
from math import sin, radians, cos, asin, sqrt
import math
import pandas as pd
from collections import Counter
import numpy as np
import sys
import pickle
from sklearn.impute import SimpleImputer
sys.path.insert(0, "lib")
import json

def GetTimeFromLine(line):
    A = line.strip("\n")  # 去空格
    B = A.split(",")  # 去每一行之间的逗号
    C = time.mktime(time.strptime(B[5], '%Y-%m-%d'))  # 年月日距1899年多少秒
    D = B[6].split(":")  # 时分秒 算成一共的秒数
    E = C + int(D[0]) * 3600 + int(D[1]) * 60 + int(D[2])  # 距离1899年过的总秒数
    return E

def haversine(lon1,lat1,lon2,lat2):
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])  # radians 角度转弧度
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a))  # 反正弦 角度
    r = 6371
    # d 直线距离
    d = r * sqrt(2 * (1-cos(c)))
    return c * r * 1000, d

def azimuth_angle(x1,y1,x2,y2):
    angle = 0
    dy = y2 - y1
    dx = x2 - x1
    if dx == 0 and dy > 0:
        angle = 0
    if dx == 0 and dy < 0:
        angle = 180
    if dy == 0 and dx > 0:
        angle = 90
    if dy == 0 and dx < 0:
        angle = 270
    if dx > 0 and dy > 0:
       angle = math.atan(dx/dy) * 180 / math.pi
    elif dx < 0 and dy>0:
       angle = 360 + math.atan(dx/dy) * 180 / math.pi
    elif dx < 0 and dy<0:
       angle = 180 + math.atan(dx/dy) * 180 / math.pi
    elif dx > 0 and dy<0:
       angle = 180 + math.atan(dx/dy) * 180 / math.pi
    return angle

def GetMaxMin3(a,list):
    if len(list) != 6:
        list.append(a)
    else:
        if a > list[3] and a not in list:
            list[3] = a
        if a < list[2] and a not in list:
            list[2] = a
    list.sort()
    return list

def calAutoCorCoe(data,u):
    n = len(data)
    c0 = 0
    c1 = 0
    for i in range(n-1):
        c0 += math.pow(data[i]-u, 2)
        c1 += (data[i]-u)*(data[i+1]-u)
    c0 += math.pow(data[n-1]-u, 2)
    if c0 == 0:
        auto = 0
    elif c0 != 0:
        auto = c1 / c0
    return auto


def cal_HCR_VCR_SR(valuelist, b):
    n = 0
    if b == 0:
        for k in range(len(valuelist)):
            if valuelist[k] > 19:
                n += 1
        n = n / len(valuelist)
    elif b == 1:
        for k in range(len(valuelist)-1):
            if valuelist[k+1] - valuelist[k+1] > 3.4:
                n += 1
        n = n / (len(valuelist)-1)
    elif b == 2:
        for k in range(len(valuelist)-1):
            if valuelist[k+1] - valuelist[k+1] < 0.26:
                n += 1
        n = n / (len(valuelist)-1)
    else:
        print("请输入正确的b值")
    return n


def st_trajectory_model(path):
    info = []
    f = pd.read_csv(path+'/'+'trajectory.csv')      ################################
    users_id = list(set(f.iloc[:, 0]))
    s_id = []
    for u in users_id:
        slice_id = list(set(f.loc[f['user_id'] == u].iloc[:, 1]))
        s_id.append(slice_id)
        for s in slice_id:
            Data = f.loc[(f['user_id'] == u) & (f['slice_id'] == s)]
            Data.to_csv(path+'/'+'traj'+str(u)+str(s)+'.csv',header=None,index=None)   ################################

    for u in range(len(users_id)):
        for s in s_id[u]:
            with open(path+'/'+'traj'+str(users_id[u])+str(s)+'.csv', 'r') as file:    ################################
                line1 = file.readline()
                a1 = line1.strip("\n")
                b1 = a1.split(",")
                f1 = time.mktime(time.strptime(b1[5], '%Y-%m-%d'))
                g1 = b1[6].split(":")
                h1 = int(g1[0]) * 3600 + int(g1[1]) * 60 + int(g1[2])
                i1 = f1 + h1

                line2 = file.readline()
                a2 = line2.strip("\n")
                b2 = a2.split(",")
                f2 = time.mktime(time.strptime(b2[5], '%Y-%m-%d'))
                g2 = b2[6].split(":")
                h2 = int(g2[0]) * 3600 + int(g2[1]) * 60 + int(g2[2])
                i2 = f2 + h2

                v_list = []
                a_list = []
                L_list = []
                t_list = []
                TurnAngel_list = []
                Sinuosity_list = []
                MaxMin_V_List = []
                MaxMin_A_List = []
                MaxMin_TA_List = []
                MaxMin_Sinuosity_List = []

                line3 = file.readline()
                while line3:
                    a3 = line3.strip("\n")
                    b3 = a3.split(",")
                    f3 = time.mktime(time.strptime(b3[5], '%Y-%m-%d'))
                    g3 = b3[6].split(":")
                    h3 = int(g3[0]) * 3600 + int(g3[1]) * 60 + int(g3[2])
                    i3 = f3 + h3

                    if i2 - i1 != 0 and i3 - i2 != 0:
                        L1, d1 = haversine(float(b1[3]), float(b1[2]), float(b2[3]), float(b2[2]))
                        L2, d2 = haversine(float(b2[3]), float(b2[2]), float(b3[3]), float(b3[2]))
                        v1 = L1 / (i2 - i1)
                        v2 = L2 / (i3 - i2)
                        a1 = (v2 - v1) / (i2 - i1)
                        Azimuth1 = azimuth_angle(float(b1[3]), float(b1[2]), float(b2[3]), float(b2[2]))
                        Azimuth2 = azimuth_angle(float(b2[3]), float(b2[2]), float(b3[3]), float(b3[2]))
                        Deflection_angle1 = Azimuth1 - Azimuth2
                        Sinuosity = (L1 / d1 if L1 != 0 and d1 != 0 else 0)

                        v_list.append(v1)
                        a_list.append(a1)
                        L_list.append(L1)
                        t_list.append(i2 - i1)
                        TurnAngel_list.append(Deflection_angle1)
                        Sinuosity_list.append(Sinuosity)

                        MaxMin_V_List = GetMaxMin3(v1, MaxMin_V_List)
                        MaxMin_A_List = GetMaxMin3(a1, MaxMin_A_List)
                        MaxMin_TA_List = GetMaxMin3(Deflection_angle1, MaxMin_TA_List)
                        MaxMin_Sinuosity_List = GetMaxMin3(Sinuosity, MaxMin_Sinuosity_List)

                    line1 = line2
                    line2 = line3
                    line3 = file.readline()

                    a1 = line1.strip("\n")
                    b1 = a1.split(",")
                    f1 = time.mktime(time.strptime(b1[5], '%Y-%m-%d'))
                    g1 = b1[6].split(":")
                    h1 = int(g1[0]) * 3600 + int(g1[1]) * 60 + int(g1[2])
                    i1 = f1 + h1

                    a2 = line2.strip("\n")
                    b2 = a2.split(",")
                    f2 = time.mktime(time.strptime(b2[5], '%Y-%m-%d'))
                    g2 = b2[6].split(":")
                    h2 = int(g2[0]) * 3600 + int(g2[1]) * 60 + int(g2[2])
                    i2 = f2 + h2

                # 补齐操作
                if len(v_list) == 0 or len(a_list) == 0 or len(TurnAngel_list) == 0 or len(Sinuosity_list) == 0:
                    continue
                else:
                    v_list.append(v_list[-1])
                    a_list.append(a_list[-1])
                    a_list.append(a_list[-1])
                    TurnAngel_list.append(TurnAngel_list[-1])
                    Sinuosity_list.append(Sinuosity_list[-1])

                # 特征计算
                meanV = sum(L_list) / sum(t_list)
                meanA = np.mean(a_list)
                meanTA = np.mean(TurnAngel_list)
                meanSinuosity = np.mean(Sinuosity_list)

                # 速度的标准差单独计算，因为是按总距离除总时间算的
                var_V = np.sqrt(((v_list - np.mean(v_list)) ** 2).sum() / (len(v_list) - 1))
                var_A = np.var(a_list, ddof=1)
                var_TA = np.var(TurnAngel_list, ddof=1)
                var_Sinuosity = np.var(Sinuosity_list, ddof=1)

                # 计算Mode，即出现在列表最多的值（出现多个值出现相同次数的情况，取最小值）
                list_value_V = list(Counter(v_list).values())
                mode_V = v_list[list_value_V.index(max(list_value_V))]
                list_value_A = list(Counter(a_list).values())
                mode_A = a_list[list_value_A.index(max(list_value_A))]
                list_value_TA = list(Counter(TurnAngel_list).values())
                mode_TA = TurnAngel_list[list_value_TA.index(max(list_value_TA))]
                list_value_Sinuosity = list(Counter(Sinuosity_list).values())
                mode_Sinuosity = Sinuosity_list[list_value_Sinuosity.index(max(list_value_Sinuosity))]

                # 最大三个值/最小三个值/值域
                if 0 < len(MaxMin_V_List) < 6:
                    continue
                if 0 < len(MaxMin_A_List) < 6:
                    continue
                if 0 < len(MaxMin_TA_List) < 6:
                    continue
                if 0 < len(MaxMin_Sinuosity_List) < 6:
                    continue

                Min1_V, Min2_V, Min3_V, Max1_V, Max2_V, Max3_V = MaxMin_V_List[0], MaxMin_V_List[1], MaxMin_V_List[
                    2], \
                                                                 MaxMin_V_List[5], MaxMin_V_List[4], MaxMin_V_List[
                                                                     3]
                Min1_A, Min2_A, Min3_A, Max1_A, Max2_A, Max3_A = MaxMin_A_List[0], MaxMin_A_List[1], MaxMin_A_List[
                    2], \
                                                                 MaxMin_A_List[5], MaxMin_A_List[4], MaxMin_A_List[
                                                                     3]
                Min1_TA, Min2_TA, Min3_TA, Max1_TA, Max2_TA, Max3_TA = MaxMin_TA_List[0], MaxMin_TA_List[1], \
                                                                       MaxMin_TA_List[
                                                                           2], MaxMin_TA_List[5], MaxMin_TA_List[4], \
                                                                       MaxMin_TA_List[3]
                Min1_Sinuosity, Min2_Sinuosity, Min3_Sinuosity = MaxMin_Sinuosity_List[0], MaxMin_Sinuosity_List[1], \
                                                                 MaxMin_Sinuosity_List[2]
                Max1_Sinuosity, Max2_Sinuosity, Max3_Sinuosity = MaxMin_Sinuosity_List[5], MaxMin_Sinuosity_List[4], \
                                                                 MaxMin_Sinuosity_List[3]

                ValueRange_V = Max1_V - Min1_V
                ValueRange_A = Max1_A - Min1_A
                ValueRange_TA = Max1_TA - Min1_TA
                ValueRange_Sinuosity = Max1_Sinuosity - Min1_Sinuosity

                # 25/75百分位数/ 分位域
                lowQua_V, upQua_V, RangeQua_V = np.percentile(np.array(v_list), 25), np.percentile(np.array(v_list),
                                                                                                   75), np.percentile(
                    np.array(v_list), 75) - np.percentile(np.array(v_list), 25)
                lowQua_A, upQua_A, RangeQua_A = np.percentile(np.array(a_list), 25), np.percentile(np.array(a_list),
                                                                                                   75), np.percentile(
                    np.array(a_list), 75) - np.percentile(np.array(a_list), 25)
                lowQua_TA, upQua_TA, RangeQua_TA = np.percentile(np.array(TurnAngel_list), 25), np.percentile(
                    np.array(TurnAngel_list), 75), np.percentile(np.array(TurnAngel_list), 75) - np.percentile(
                    np.array(TurnAngel_list), 25)
                lowQua_Sinuosity, upQua_Sinuosity, RangeQua_Sinuosity = np.percentile(np.array(Sinuosity_list),
                                                                                      25), np.percentile(
                    np.array(Sinuosity_list), 75), np.percentile(np.array(Sinuosity_list), 75) - np.percentile(
                    np.array(Sinuosity_list), 25)

                # 偏度Skewness / 峰度Kurtosis / 变异系数coefficient of variation /自相关系数ACC
                Skew_V = pd.DataFrame({'V': v_list}).skew(axis=0).V
                Skew_A = pd.DataFrame({'A': a_list}).skew(axis=0).A
                Skew_TA = pd.DataFrame({'TA': TurnAngel_list}).skew(axis=0).TA
                Skew_Sinuosity = pd.DataFrame({'Sinuosity': Sinuosity_list}).skew(axis=0).Sinuosity

                Kurt_V = pd.DataFrame({'V': v_list}).kurt(axis=0).V
                Kurt_A = pd.DataFrame({'A': a_list}).kurt(axis=0).A
                Kurt_TA = pd.DataFrame({'TA': TurnAngel_list}).kurt(axis=0).TA
                Kurt_Sinuosity = pd.DataFrame({'Sinuosity': Sinuosity_list}).kurt(axis=0).Sinuosity

                CV_V = meanV / var_V
                CV_A = meanA / var_A
                CV_TA = meanTA / var_TA
                CV_Sinuosity = meanSinuosity / var_Sinuosity

                AutoCC_V = calAutoCorCoe(v_list, meanV)
                AutoCC_A = calAutoCorCoe(a_list, meanA)
                AutoCC_TA = calAutoCorCoe(TurnAngel_list, meanTA)
                AutoCC_Sinuosity = calAutoCorCoe(Sinuosity_list, meanSinuosity)

                # 方向变化率HCR / 停止率SR / 速度变化率VCR / 轨迹length
                HCR = cal_HCR_VCR_SR(TurnAngel_list, 0)
                SR = cal_HCR_VCR_SR(v_list, 2)
                VCR = cal_HCR_VCR_SR(v_list, 1)
                length = len(v_list)
                X = []
                X.append([meanV, meanA, meanTA, meanSinuosity, var_V, var_A, var_TA, var_Sinuosity, mode_V, mode_A,
                          mode_TA, mode_Sinuosity, Min1_V, Min2_V, Min3_V, Max1_V, Max2_V, Max3_V,
                          Min1_A, Min2_A, Min3_A, Max1_A, Max2_A, Max3_A,
                          Min1_TA, Min2_TA, Min3_TA, Max1_TA, Max2_TA, Max3_TA,
                          Min1_Sinuosity, Min2_Sinuosity, Min3_Sinuosity, Max1_Sinuosity, Max2_Sinuosity,
                          Max3_Sinuosity,
                          ValueRange_V, ValueRange_A, ValueRange_TA, ValueRange_Sinuosity,
                          lowQua_V, upQua_V, RangeQua_V,
                          lowQua_A, upQua_A, RangeQua_A,
                          lowQua_TA, upQua_TA, RangeQua_TA,
                          lowQua_Sinuosity, upQua_Sinuosity, RangeQua_Sinuosity,
                          Skew_V, Skew_A, Skew_TA, Skew_Sinuosity,
                          Kurt_V, Kurt_A, Kurt_TA, Kurt_Sinuosity,
                          CV_V, CV_A, CV_TA, CV_Sinuosity,
                          AutoCC_V, AutoCC_A, AutoCC_TA, AutoCC_Sinuosity,
                          HCR, SR, VCR, length])

                X = SimpleImputer().fit_transform(X)
                X_test = np.array(X)
                with open(path+"/test.pkl", "rb") as f:     ################################
                    gc = pickle.load(f)
                y_pred = gc.predict(X_test)
                if y_pred[0] == 0:
                    mode = 'walk'
                if y_pred[0] == 1:
                    mode = 'bike'
                if y_pred[0] == 2:
                    mode = 'bus'
                if y_pred[0] == 3:
                    mode = 'car'
                if y_pred[0] == 4:
                    mode = 'subway'
                if y_pred[0] == 5:
                    mode = 'train'
                if y_pred[0] == 6:
                    mode = 'hybrid mode'

                tmp_result = {'user_ID':str(users_id[u]),'slice_ID':str(s),"mode":mode,"meanV":round(meanV,4), "meanA":round(meanA,4), "meanTA":round(meanTA,4), "meanSinuosity":round(meanSinuosity,4), "var_V":round(var_V,4), "var_A":round(var_A,4), "var_TA":round(var_TA,4), "var_Sinuosity":round(var_Sinuosity,4), "mode_V":round(mode_V,4), "mode_A":round(mode_A,4),
                          "mode_TA":round(mode_TA,4), "mode_Sinuosity":round(mode_Sinuosity,4), "Min1_V":round(Min1_V,4), "Min2_V":round(Min2_V,4), "Min3_V":round(Min3_V,4), "Max1_V":round(Max1_V,4), "Max2_V":round(Max2_V,4), "Max3_V":round(Max3_V,4),
                          "Min1_A":round(Min1_A,4), "Min2_A":round(Min2_A,4), "Min3_A":round(Min3_A,4), "Max1_A":round(Max1_A,4), "Max2_A":round(Max2_A,4), "Max3_A":round(Max3_A,4),
                          "Min1_TA":round(Min1_TA,4), "Min2_TA":round(Min2_TA,4), "Min3_TA":round(Min3_TA,4), "Max1_TA":round(Max1_TA,4), "Max2_TA":round(Max2_TA,4), "Max3_TA":round(Max3_TA,4),
                          "Min1_Sinuosity":round(Min1_Sinuosity,4), "Min2_Sinuosity":round(Min2_Sinuosity,4), "Min3_Sinuosity":round(Min3_Sinuosity,4), "Max1_Sinuosity":round(Max1_Sinuosity,4), "Max2_Sinuosity":round(Max2_Sinuosity,4),
                          "Max3_Sinuosity":round(Max3_Sinuosity,4),
                          "ValueRange_V":round(ValueRange_V,4), "ValueRange_A":round(ValueRange_A,4), "ValueRange_TA":round(ValueRange_TA,4), "ValueRange_Sinuosity":round(ValueRange_Sinuosity,4),
                          "lowQua_V":round(lowQua_V,4), "upQua_V":round(upQua_V,4), "RangeQua_V":round(RangeQua_V,4),
                          "lowQua_A":round(lowQua_A,4), "upQua_A":round(upQua_A,4), "RangeQua_A":round(RangeQua_A,4),
                          "lowQua_TA":round(lowQua_TA,4), "upQua_TA":round(upQua_TA,4), "RangeQua_TA":round(RangeQua_TA,4),
                          "lowQua_Sinuosity":round(lowQua_Sinuosity,4), "upQua_Sinuosity":round(upQua_Sinuosity,4), "RangeQua_Sinuosity":round(RangeQua_Sinuosity,4),
                          "Skew_V":round(Skew_V,4), "Skew_A":round(Skew_A,4), "Skew_TA":round(Skew_TA,4), "Skew_Sinuosity":round(Skew_Sinuosity,4),
                          "Kurt_V":round(Kurt_V,4), "Kurt_A":round(Kurt_A,4), "Kurt_TA":round(Kurt_TA,4), "Kurt_Sinuosity":round(Kurt_Sinuosity,4),
                          "CV_V":round(CV_V,4), "CV_A":round(CV_A,4), "CV_TA":round(CV_TA,4), "CV_Sinuosity":round(CV_Sinuosity,4),
                          "AutoCC_V":round(AutoCC_V,4), "AutoCC_A":round(AutoCC_A,4), "AutoCC_TA":round(AutoCC_TA,4), "AutoCC_Sinuosity":round(AutoCC_Sinuosity,4),
                          "HCR":round(HCR,4), "SR":SR, "VCR":VCR, "length":length}

                info.append(tmp_result)

    result = json.dumps(info)
    return result

if __name__ == '__main__':
    path = r'E:/building/wpy' ################################
    result = st_trajectory_model(path)
    print(result)


