import numpy as np
import random as rd
import csv
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']


# 计算两点间的距离，只计算13个数据（不包括红酒品类）
def get_distance(cent, data):
    d = 0
    for i in range(1, 14):
        d += pow(cent[i]-data[i], 2)
    return d


# 获取所有最近簇质心为j的点
def get_points(data_nums, data, j, kmeans_mat):
    points = []
    for i in range(data_nums):
        if kmeans_mat[i, 0] == j + 1:
            points.append(data[i])
    return points


def get_accpoints(data_nums, data, j):
    point = []
    data = np.array(data)
    for i in range(data_nums):
        if int(data[i, 0]) == j + 1:    # 获取所有品类=j+1的点
            point.append(data[i])
    return point


# 计算所有点到其簇质心的平方和
def get_sse(data_nums, kmeans_mat):
    sse_single = [0, 0, 0]
    for i in range(data_nums):
        sse_single[int(kmeans_mat[i, 0])-1] += kmeans_mat[i, 1]
    sse = sum(sse_single)
    print(f"每个cluster中的点到其簇质心的距离平方和:{sse_single}")
    print(f"所有点到其簇质心的距离平方和:{sse}")
    return sse


def get_acc(data_nums, data, k, kmeans_mat):
    cluster1 = kmeans_mat[0, 0]       # 第一个品类的红酒的簇质心
    cluster3 = kmeans_mat[data_nums-1, 0]     # 第三个品类的红酒的簇质心
    cluster2 = max(3-cluster1, 3-cluster3, 6-(cluster1+cluster3))   # 第二个品类的红酒的簇质心
    centroid = [cluster1, cluster2, cluster3]   # k-means后三个品类的红酒的簇质心
    acc_single = []
    for i in range(k):
        points = get_points(data_nums, data, centroid[i]-1, kmeans_mat)
        acc_points = get_accpoints(data_nums, data, i)
        acc_temp = min(len(points) / len(acc_points), len(acc_points) / len(points))
        acc_single.append(acc_temp)
    acc_average = sum(acc_single) / 3
    print(f"每个品类的准确度:{acc_single}")
    print(f"平均准确度:{acc_average}")
    return acc_average


if __name__ == '__main__':
    with open("./normalizedwinedata.csv", 'r') as fr:
        rows = csv.reader(fr)  # 一行作为一个字符串列表
        data = []
        for row in rows:    # 循环访问每个字符串列表（每行），红酒品类也要保存，用于计算准确率acc
            row = list(map(float, row))
            data.append(row)
        # print(data)

    # 随机生成3个簇质心
    k = 3  # cluster的数量
    centroids = []  # 簇质心
    for i in range(k):
        centroid = []
        for j in range(14):     # 1个品类+13维数据
            centroid.append(rd.random())   # 随机生成簇质心：random()返回一个随机生成的浮点数，范围在[0,1)之间
        centroids.append(centroid)
    # print(centroids)

    data_nums = len(data)  # 多少个数据，每个数据是一个13维的点
    kmeans_mat = np.mat(np.zeros((data_nums, 2)))  # 创建m行2列的矩阵，每列分别表示每个点的最近簇质心和到该簇质心的距离

    change = True
    while change:
        change = False
        # 对每个点
        for i in range(data_nums):
            min_distance = 100000000.0
            min_centroid = -1
            # print(f"修正前，{i}到三个簇距离：")
            for j in range(k):  # 计算这个点到3个簇质心的距离
                # print(j)
                # print(centroids[j])
                distance = get_distance(centroids[j], data[i])
                if distance < min_distance:
                    min_centroid = j + 1  # 记录最近簇质心
                    min_distance = distance    # 以及到该簇质心的距离
            if kmeans_mat[i, 0] != min_centroid or kmeans_mat[i, 1] != min_distance:     # 如果该点的最近簇质心改变，或者到最近簇质心的距离改变
                kmeans_mat[i, :] = min_centroid, min_distance      # 更新矩阵
                change = True   # 继续循环，直到稳定（矩阵中数值不变）
        # 修正簇质心
        # print("修正后")
        for j in range(k):
            # 获取所有簇质心为j+1的点
            # print(f"簇质心编号{j+1}")
            points = get_points(data_nums, data, j, kmeans_mat)
            # print(f"点数{len(points)}")
            # print(f"簇中的点{points}")
            centroids[j] = np.mean(points, axis=0)  # axis=0计算列的均值
            # print(f"簇质心{centroids[j]}")

    SSE = get_sse(data_nums, kmeans_mat)    # 计算SSE
    Acc = get_acc(data_nums, data, k, kmeans_mat)   # 计算acc

    # 效果展示图，在聚类之后，任选两个维度，以三种不同的颜色对自己聚类的结果进行标注，最终以二维平面中点图的形式来展示三个质心和所有的样本点
    cValues = ['r', 'g', 'b']    # 散点颜色 red/green/blue
    # 任选两个维度
    x = 6
    y = 7

    fig = plt.figure("Sample Diagram")
    ax1 = fig.add_subplot()
    ax1.set_title(f"SSE={format(SSE, '.3f')}  Acc={format(Acc, '.3f')}")    # 设置标题

    # 绘制所有样本点
    for j in range(k):
        points = get_points(data_nums, data, j, kmeans_mat)
        points = np.array(points)
        ax1.scatter(points[:, x], points[:, y], c=cValues[j])
    # 绘制簇质心
    centroids = np.array(centroids)
    ax1.scatter(centroids[:, x], centroids[:, y], marker='^', s=100, c='black')
    # 显示样本图
    plt.xlabel(f"x={x}")
    plt.ylabel(f"y={y}")
    plt.show()

    # 输出结果矩阵
    with open("result.txt", 'w') as fw:
        for i in range(data_nums):
            fw.write(f"{int(kmeans_mat[i, 0])} ,{kmeans_mat[i, 1]}\n")