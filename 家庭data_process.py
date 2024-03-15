'''
数据格式：
前3个数值 xx xx xx 对应 小时 分钟 秒
2个数值 xx xx xx xx 对应 2个天线RSSI
1个数值 x 对应 mcs
2个数值 x x x x 对应 2个天线gain
500个复数值对应 Tx1-Rx1的 H
500个复数值对应 Tx1-Rx2的 H

依次重复。
'''
import numpy as np
# 打开.txt文件以读取数据
with open('F:/HUAWEI-WiFi/1031data/家庭场景1-房间A/房间A/csi_2023_10_31_5.txt', 'r') as file:
    # 逐行读取文件内容
    data = file.readlines()
    print(len(data)) # 数据有1行(没有换行了)
# 关闭文件
file.close()

for line in data:
    data1 = line.strip().split()
    print(len(data1))
n = len(data1) // 1008
# 现在，'data1'变量包含了文件中的所有行数据
time = np.zeros((n, 3))
mcs = np.zeros((n, 1))
Rx_RSSI = np.zeros((n, 2))
Rx_gain = np.zeros((n, 2))
H_Rx1 = np.zeros((n, 500), dtype=complex)
H_Rx2 = np.zeros((n, 500), dtype=complex) 
# H_Rx3 = np.zeros((n, 64), dtype=complex)
# H_Rx4 = np.zeros((n, 64), dtype=complex)

print(data1[0:1008])
# 以换行符分割数据
# 每一行有1008个数据，1008 = 3(time) + 2(two RSSI) + 1(mcs) + 2(two antenna gain) + 500 * 2 (four H_Rx)
t = 1008
for i in range(n):
    time[i,:] = data1[0+i*t:3+i*t]
    Rx_RSSI[i,:] = data1[3+i*t:5+i*t]
    mcs[i,:] = data1[6+i*t]
    Rx_gain[i,:] = data1[6+i*t:8+i*t]
    H_Rx1[i,:] = [complex(comp_str.replace("i", "j")) for comp_str in data1[8+i*t:508+i*t]]
    H_Rx2[i,:] = [complex(comp_str.replace("i", "j")) for comp_str in data1[508+i*t:1008+i*t]]
    # H_Rx3[i,:] = [complex(comp_str.replace("i", "j")) for comp_str in data1[140+i*268:204+i*268]]
    # H_Rx4[i,:] = [complex(comp_str.replace("i", "j")) for comp_str in data1[204+i*268:268+i*268]] 


print(time[0:10, :])
print(Rx_RSSI[0:10,:])
print(mcs[0:10, :])
print(Rx_gain[0:10, :])
print(H_Rx1[0:2,:])
print(H_Rx2[0:2,:])
# print(H_Rx3[0:2,:])
# print(H_Rx4[0:2,:])

