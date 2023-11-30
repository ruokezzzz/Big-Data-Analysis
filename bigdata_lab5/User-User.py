import pandas as pd
import numpy as np
import csv

global user_movie


# 构建效用矩阵
def get_utility_matrix():
    df = pd.read_csv('train_set.csv')  # timestamp不用读
    global user_movie
    user_movie = df.pivot(index='userId', columns='movieId', values='rating')
    # # 调试
    # print(user_movie)
    # print(type(user_movie))     # DataFrame
    # print(user_movie.loc[5])
    # print(type(user_movie.loc[5]))  # Series
    # print(user_movie.loc[5][3])     # 4.0
    # print(type(user_movie.loc[5][3]))   # numpy.float64
    # # 用户1最相似的30个用户
    # similar_k_users = user_movie.drop(1).index.to_series().apply(pearson, args=(1,)).nlargest(30)
    # print(similar_k_users)
    # print(type(similar_k_users))
    # # 用户2和3的共同电影评分Series
    # filter = user_movie.loc[2].notnull() & user_movie.loc[3].notnull()
    # x = user_movie.loc[2, filter]
    # y = user_movie.loc[3, filter]
    # print(x)
    # print(type(x))
    # print(y)
    # print(type(y))


# pearson相似度计算
def pearson(user_id1, user_id2):
    # 首先找到这两个用户都看过的电影
    filter = user_movie.loc[user_id1].notnull() & user_movie.loc[user_id2].notnull()
    x = user_movie.loc[user_id1, filter]
    y = user_movie.loc[user_id2, filter]
    divisor = (sum((x-x.mean())**2)*sum((y-y.mean())**2))**0.5
    try:
        value = sum((x - x.mean()) * (y - y.mean())) / divisor
    except ZeroDivisionError:   # 分母为0，说明这两个用户没有共同看过的电影，毫不相关，pearson=0
        value = 0
    return value


# 预测目标用户user_id对目标电影movie_id的评分
def usercf(user_id, movie_id, k=30):
    # 首先找到与用户user_id最相似的k个用户
    # 方法是在效用矩阵中对除了user_id的每个user，计算他与user_id的Pearson相似度，取最大的k个users
    # similar_k_users索引为userId，值为Pearson相似度
    similar_k_users = user_movie.drop(user_id).index.to_series().apply(pearson, args=(user_id,)).nlargest(k)
    similar_k_userid = similar_k_users.index
    # ans = []
    pearson_sum = 0     # 预测分数计算公式的分母
    part_sum = 0        # 预测分数计算公式的分子
    for similar_id in similar_k_userid:  # 对每个相似用户
        # 找到预测用户没打分且该相似用户打分了的电影tmp，索引为movieId，值为分数
        filter = user_movie.loc[user_id].isnull() & user_movie.loc[similar_id].notnull()
        tmp = user_movie.loc[similar_id, filter]
        if movie_id in tmp.index:   # 如果目标电影在这些电影中，则记录评分，用于后续预测
            pearson_sum += similar_k_users[similar_id]                  # 分母累加：该相似用户与指定用户的相似度
            part_sum += tmp[movie_id] * similar_k_users[similar_id]     # 分子累加：分数×相似度
    # if len(ans) == 0:   # 极端情况：所有相似用户都没看目标电影,取目标用户打分平均值
    if part_sum == 0:  # 极端情况：所有相似用户都没看目标电影,取目标用户打分平均值
        result = user_movie.loc[user_id].mean()
    else:
        # result = ans.mean()
        result = part_sum / pearson_sum     # 加权平均，权重为相似度
    return result


# 推荐函数
def recommend(user_id, movie_id, similar_k_users):
    similar_k_userid = similar_k_users.index
    pearson_sum = 0     # 预测分数计算公式的分母
    part_sum = 0        # 预测分数计算公式的分子
    for similar_id in similar_k_userid:  # 对每个相似用户
        # 找到预测用户没打分且该相似用户打分了的电影tmp，索引为movieId，值为分数
        filter = user_movie.loc[user_id].isnull() & user_movie.loc[similar_id].notnull()
        tmp = user_movie.loc[similar_id, filter]
        if movie_id in tmp.index:   # 如果这些电影中包含目标电影
            pearson_sum += similar_k_users[similar_id]              # 分母累加：该相似用户与指定用户的相似度
            part_sum += tmp[movie_id]*similar_k_users[similar_id]   # 分子累加：分数×相似度
    if part_sum == 0:  # 极端情况：所有相似用户都没看目标电影,取目标用户打分平均值
        result = user_movie.loc[user_id].mean()
    else:
        result = part_sum / pearson_sum     # 加权平均，权重为相似度
    return result


if __name__ == '__main__':
    get_utility_matrix()

    # 测试
    # 读取测试集
    testDf = pd.read_csv('test_set.csv')
    testUserIds = testDf['userId']
    testMovieIds = testDf['movieId']
    testRatings = testDf['rating']
    # 计算
    predict_score = []
    sseList = []
    for i in range(len(testUserIds)):
        predict_score.append(usercf(testUserIds[i], testMovieIds[i]))
        sseList.append((predict_score[i] - testRatings[i])**2)
    # 文件输出
    with open("./result1.txt", "w") as fw:
        fw.write("userId, movieId, real rating, predict rating, SSE\n")
        for i in range(len(testUserIds)):
            fw.write(f"{testUserIds[i]}, {testMovieIds[i]}, {testRatings[i]}, {predict_score[i]}, {sseList[i]}\n")
        sse = sum(sseList)
        fw.write(f"SSE: {sse}")
    # 控制台输出
    print("%-8s" % "userId", "%-8s" % "movieId", "%-8s" % "实际评分", "%-8s" % "预测评分", "%-8s" % "SSE")
    for i in range(len(testUserIds)):
        print("%-8d" % testUserIds[i], "%-8d" % testMovieIds[i], "%-10f" % testRatings[i],
              "%-10f" % predict_score[i], "%-10f" % sseList[i])
    sse = sum(sseList)
    print("SSE: ", sse)

    # # 推荐
    # # userId, n = map(int, input("输入userID和推荐的电影个数n:").split())
    # userId = 4
    # n = 5
    # # 获取相似用户
    # similar_k_users = user_movie.drop(userId).index.to_series().apply(pearson, args=(userId,)).nlargest(30)
    # # 获取电影id到电影名字的映射
    # df = pd.read_csv('movies.csv')
    # movieIds = df["movieId"]
    # movie_titles = df["title"]
    # movie_dict = {}
    # for i in range(len(movieIds)):
    #     movie_dict[movieIds[i]] = movie_titles[i]
    # # 对所有指定用户没看过的电影进行预测
    # pre_dict = {}
    # for movie_id in movie_dict:
    #     if movie_id not in user_movie.loc[userId, user_movie.loc[userId].notnull()].index:
    #         pre_dict[movie_id] = recommend(userId, movie_id, similar_k_users)
    #         # print(f"{movie_id}: {pre_dict[movie_id]}")
    # # 对评分排序，输出前n名电影
    # ans = sorted(pre_dict.items(), key=lambda data: data[1], reverse=True)
    # for i in range(n):
    #     print(f"{movie_dict[ans[i][0]]}: {ans[i][1]}")
