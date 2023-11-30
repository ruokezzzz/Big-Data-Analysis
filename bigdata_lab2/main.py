import pandas as pd
import numpy as np


# # 显示所有列
# pd.set_option('display.max_columns', None)
# # 显示所有行
# pd.set_option('display.max_rows', None)
# # 设置value的显示长度
# pd.set_option('max_colwidth', 100)
# # 设置1000列时才换行
# pd.set_option('display.width', 1000)


N = 513     # 人数(page nodes)

# 构造邻接矩阵M
M = np.zeros((N, N))
df = pd.read_csv("./sent_receive.csv")  # 打开数据文件
for (sent_id, part_1) in list(df.groupby('sent_id')):
    receivers = []  # 获取每个sender对应的receivers
    for (receiver_id, part_2) in list(part_1.groupby('receive_id')):
        receivers.append(int(receiver_id))
    out_degree = len(receivers)     # 计算这个sender的出度，即receivers的数量
    for item in receivers:
        M[item-1, sent_id-1] = 1 / out_degree

r = np.ones((N, 1)) / N     # 初始化r向量
Beta = 0.9  # 进阶版考虑加入teleport β，用以对概率转移矩阵进行修正，解决dead ends和spider trap的问题
A = M*Beta + np.ones((N, N))/N * (1-Beta)   # 矩阵A


# r' = A·r
while 1:
    new_r = np.dot(A, r)
    error = (new_r - r)**2
    # 判断误差
    if np.sum(error) < 1e-10:
        break
    r = new_r

# 文件输出
with open("./result.txt", "w") as fw:
    for i in range(len(r)):
        fw.write(f"{i+1} : {float(r[i])}\n")