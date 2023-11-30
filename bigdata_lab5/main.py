import pandas as pd
from numpy import *
import numpy as np


def read_movies():
    moviedf = pd.read_csv('movies.csv')
    movie_genres = list(moviedf['genres'])
    moviesIds = moviedf['movieId']
    # 创建电影类别到电影的映射 {类型：[电影1，电影2...]}
    genre_movies_dict = {}
    for i in range(len(moviesIds)):     # 遍历每个电影的每个类型
        genres = movie_genres[i].split('|')     # 提取每个电影的所有类型，以'|'分隔生成字符串列表
        movie_genres[i] = genres    # movie_genres列表记录每个电影的类型，计算tf值时会用到。
        for genre in genres:    # 遍历该电影的每个类型
            if genre != '(no genres listed)':
                genre_movies_dict.setdefault(genre, []).append(moviesIds[i])    # 在以这个类型为键的值列表中添加这个电影id
    # 给类型编号(0-18)，方便后面构造特征矩阵（作为矩阵的列号）
    genre_number = {}
    i = 0
    for genre in genre_movies_dict:
        genre_number[genre] = i
        i += 1
    # 给电影Id编号(0-9124)（因为电影id不是按顺序的），方便后续构造特征矩阵（作为矩阵的行号）
    movie_number = {}
    i = 0
    for id in moviesIds:
        movie_number[id] = i
        i += 1
    return genre_movies_dict, genre_number, movie_number, movie_genres


# 每个用户看的电影以及打分情况，一共671个人，user_movies=[0,{电影：评分}，{电影，评分}]
def get_user_movies():
    traindf = pd.read_csv('train_set.csv')
    user_ids = traindf['userId']
    movie_ids = traindf['movieId']
    ratings = traindf['rating']
    user_movies = [0]*672   # 记录所有用户的影评字典，0号元素不用，1号元素为用户1的影评字典，以此类推
    user_movie_dict = {}    # 每个用户的影评字典，键为用户所看电影，值为评分
    for user in range(len(user_ids)):   # 遍历每个<用户,电影,评分>组
        user_movie_dict[movie_ids[user]] = ratings[user]    # 更新该用户的影评字典
        # 遍历到当前用户看的最后一个电影，第一个判断条件是为了防止遍历到最后一行的时候user_ids[user+1]越界
        if (user != len(user_ids) - 1) and (user_ids[user] != user_ids[user+1]):
            user_movies[user_ids[user]] = user_movie_dict   # 将该用户的影评字典添加到user_movies列表中
            user_movie_dict = {}    # 清空，准备读取下一个用户的影评字典
    user_movies[671] = user_movie_dict
    return user_movies


# 得到关于电影与特征值的n(电影个数)*m(特征值个数)的tf-idf特征矩阵
def get_tfidf_matrix(genre_movies_dict, genre_number, movie_number, movie_genres):
    tfidf_matrix = np.zeros((9125, 19))
    for genre in genre_movies_dict:
        j = genre_number[genre]     # 获取类型编号（作为列号）
        idf = math.log(9125 / len(genre_movies_dict[genre]), 10)   # 计算idf值
        for movie in genre_movies_dict[genre]:    # 对这个类型的所有电影
            i = movie_number[movie]                   # 获取电影编号（作为行号）
            tf = 1 / len(movie_genres[i])      # 计算tf值
            tfidf_matrix[i, j] = idf * tf       # wij = tf*idf
    return tfidf_matrix


# 用余弦相似度的计算方法，得到相似度矩阵
def get_similarity_matrix(tfidf_matrix):
    n = len(tfidf_matrix)   # 矩阵相当于9125个行向量
    for i in range(n):      # 对矩阵中的每个向量，将其转换成a/|a|
        a = np.dot(tfidf_matrix[i], tfidf_matrix[i])
        if a != 0:  # 不是零向量
            tfidf_matrix[i] = tfidf_matrix[i] / math.sqrt(a)
    similarity_matrix = np.dot(tfidf_matrix, tfidf_matrix.T)
    return similarity_matrix


# 获取01矩阵：如果电影存在某特征值，则特征值为1，不存在则为0
def get_01_matrix(genre_movies_dict, genre_number, movie_number):
    zero_one_matrix = np.zeros((9125, 19))
    for genre in genre_movies_dict:
        j = genre_number[genre]
        for Id in genre_movies_dict[genre]:
            i = movie_number[Id]
            zero_one_matrix[i, j] = 1
    return zero_one_matrix


# 采用minhash算法对特征矩阵(01)进行降维处理，从而得到相似度矩阵，注意minhash采用jaccard方法计算相似度
def minhash(zero_one_matrix):
    # 转置：19 * 9125
    zero_one_matrix = zero_one_matrix.T
    # h1(x) = (x+1)mod19, h2(x) = (2x+1)mod19, h3(x) = (3x+1)mod19, h4(x) = (4x+1)mod19, h5(x) = (5x+1)mod19
    # hash_matrix[i, j]:hi+1(x)随机映射的第j个值
    hash_matrix = np.zeros((5, 19))
    for j in range(len(hash_matrix[0])):
        hash_matrix[0, j] = (j + 1) % 19
        hash_matrix[1, j] = (2 * j + 1) % 19
        hash_matrix[2, j] = (3 * j + 1) % 19
        hash_matrix[3, j] = (4 * j + 1) % 19
        hash_matrix[4, j] = (5 * j + 1) % 19
    # 构造签名矩阵
    signature_matrix = np.full((5, 9125), inf)
    for i in range(len(zero_one_matrix)):
        for j in range(len(zero_one_matrix[i])):
            if zero_one_matrix[i][j] == 1:
                if hash_matrix[0, i] < signature_matrix[0, j]:
                    signature_matrix[0, j] = hash_matrix[0, i]
                if hash_matrix[1, i] < signature_matrix[1, j]:
                    signature_matrix[1, j] = hash_matrix[1, i]
                if hash_matrix[2, i] < signature_matrix[2, j]:
                    signature_matrix[2, j] = hash_matrix[2, i]
                if hash_matrix[3, i] < signature_matrix[3, j]:
                    signature_matrix[3, j] = hash_matrix[3, i]
                if hash_matrix[4, i] < signature_matrix[4, j]:
                    signature_matrix[4, j] = hash_matrix[4, i]
    # 计算jaccard相似度
    signature_matrix = signature_matrix.T
    similarity_matrix = np.zeros((9125, 9125))
    for i in range(9125):
        similarity_matrix[i, i] = 1
        for j in range(i, 9125):
            sx = set(signature_matrix[i])
            sy = set(signature_matrix[j])
            similarity_matrix[i, j] = float(len(sx.intersection(sy))) / float(len(sx.union(sy)))
            similarity_matrix[j, i] = similarity_matrix[i, j]
    return similarity_matrix


def minhash1(zero_one_matrix):
    # 转置：19 * 9125
    zero_one_matrix = zero_one_matrix.T
    # h1(x) = (x+1)mod19, h2(x) = (2x+1)mod19, h3(x) = (3x+1)mod19, h4(x) = (4x+1)mod19, h5(x) = (5x+1)mod19
    # hash_matrix[i, j]:hi+1(x)随机映射的第j个值
    hash_matrix = np.zeros((8, 19))
    for j in range(len(hash_matrix[0])):
        hash_matrix[0, j] = (j + 1) % 19
        hash_matrix[1, j] = (2 * j + 1) % 19
        hash_matrix[2, j] = (3 * j + 1) % 19
        hash_matrix[3, j] = (4 * j + 1) % 19
        hash_matrix[4, j] = (5 * j + 1) % 19
        hash_matrix[5, j] = (6 * j + 1) % 19
        hash_matrix[4, j] = (7 * j + 1) % 19
        hash_matrix[5, j] = (8 * j + 1) % 19
    # 构造签名矩阵
    signature_matrix = np.full((8, 9125), inf)
    for i in range(len(zero_one_matrix)):
        for j in range(len(zero_one_matrix[i])):
            if zero_one_matrix[i][j] == 1:
                if hash_matrix[0, i] < signature_matrix[0, j]:
                    signature_matrix[0, j] = hash_matrix[0, i]
                if hash_matrix[1, i] < signature_matrix[1, j]:
                    signature_matrix[1, j] = hash_matrix[1, i]
                if hash_matrix[2, i] < signature_matrix[2, j]:
                    signature_matrix[2, j] = hash_matrix[2, i]
                if hash_matrix[3, i] < signature_matrix[3, j]:
                    signature_matrix[3, j] = hash_matrix[3, i]
                if hash_matrix[4, i] < signature_matrix[4, j]:
                    signature_matrix[4, j] = hash_matrix[4, i]
                if hash_matrix[5, i] < signature_matrix[5, j]:
                    signature_matrix[5, j] = hash_matrix[5, i]
                if hash_matrix[6, i] < signature_matrix[6, j]:
                    signature_matrix[6, j] = hash_matrix[6, i]
                if hash_matrix[7, i] < signature_matrix[7, j]:
                    signature_matrix[7, j] = hash_matrix[7, i]
    # 计算jaccard相似度
    signature_matrix = signature_matrix.T
    similarity_matrix = np.zeros((9125, 9125))
    for i in range(9125):
        similarity_matrix[i, i] = 1
        for j in range(i, 9125):
            sx = set(signature_matrix[i])
            sy = set(signature_matrix[j])
            similarity_matrix[i, j] = float(len(sx.intersection(sy))) / float(len(sx.union(sy)))
            similarity_matrix[j, i] = similarity_matrix[i, j]
    return similarity_matrix


'''
    根据指定的userId,获取已打分的movieId和rating
    根据相似度矩阵获取预测电影和已打分的电影的相似度
    如果当前预测电影与某已打分的电影的相似度大于0，加入计算集合
    将计算集合中的电影根据公式对当前预测的电影进行打分
'''


def get_score(userId, predict_movieId, user_movies, similarity_matrix, movie_number):
    # 计算集合
    compute_movies = {}    # {与目标电影相似度大于0的已看电影：评分}
    for movieId in user_movies[userId]:     # 遍历当前所有已打分的电影Id
        # 如果该以打分电影与目标电影的相似度大于0，加入计算合计
        if similarity_matrix[movie_number[movieId], movie_number[predict_movieId]] > 0:
            compute_movies[movieId] = user_movies[userId][movieId]
    # 对当前预测电影进行打分
    sim_sum = 0
    part_sum = 0
    sum = 0
    # 如果没有与预测电影相似度大于零的电影，取目标用户评分的均值
    if len(compute_movies) == 0:
        for movieId in user_movies[userId]:
            sum += user_movies[userId][movieId]
        score = sum / len(user_movies[userId])
    else:
        for movieId in compute_movies:
            sim = similarity_matrix[movie_number[movieId], movie_number[predict_movieId]]
            sim_sum += sim      # 分母累加：相似度
            part_sum += compute_movies[movieId] * sim      # 分子累加：相似度*分数
        score = part_sum / sim_sum
    return score


if __name__ == '__main__':
    # 加载movie.csv的数据
    genre_movies_dict, genre_number, movie_number, movie_genres = read_movies()
    # 获取训练集的user_movies列表,user_movies[i]存储了userId=i的用户对电影的打分情况
    user_movies = get_user_movies()

    # # 得到tf_idf矩阵
    # tfidf_matrix = get_tfidf_matrix(genre_movies_dict, genre_number, movie_number, movie_genres)
    # # 用余弦相似度的计算方法，得到电影之间的相似度矩阵
    # similarity_matrix = get_similarity_matrix(tfidf_matrix)

    # 得到01矩阵
    zero_one_matrix = get_01_matrix(genre_movies_dict, genre_number, movie_number)
    print("01 ok")
    # 用minhash
    similarity_matrix = minhash1(zero_one_matrix)
    print("minhash ok")

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
        predict_score.append(get_score(testUserIds[i], testMovieIds[i], user_movies, similarity_matrix, movie_number))
        sseList.append((predict_score[i] - testRatings[i])**2)
    # 控制台输出
    print("%-8s" % "userId", "%-8s" % "movieId", "%-8s" % "正确评分", "%-8s" % "预测评分", "%-8s" % "SSE")
    for i in range(len(testUserIds)):
        print("%-8d" % testUserIds[i], "%-8d" % testMovieIds[i],
              "%-10f" % testRatings[i], "%-10f" % predict_score[i], "%-10f" % sseList[i])
    sse = sum(sseList)
    print("SSE: ", sse)

    # # 给指定用户userID推荐k个电影
    # # userId, k = map(int, input("输入userID和推荐的电影个数k:").split())
    # userId = 4
    # k = 5
    # pre_dict = {}
    # 
    # df = pd.read_csv('movies.csv')
    # movieIds = df["movieId"]
    # movie_titles = df["title"]
    # movie_dict = {}
    # for i in range(len(movieIds)):
    #     movie_dict[movieIds[i]] = movie_titles[i]
    # 
    # for movie_id in movie_dict:
    #     if movie_id not in user_movies[userId]:
    #         pre_dict[movie_id] = get_score(userId, movie_id, user_movies, similarity_matrix, movie_number)
    # 
    # ans = sorted(pre_dict.items(), key=lambda data: data[1], reverse=True)
    # for i in range(k):
    #     print(f"{movie_dict[ans[i][0]]}: {ans[i][1]}")
