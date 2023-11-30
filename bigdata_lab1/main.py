from concurrent import futures


# 读取一个数据文件,map+combine,输出map文件
def mapper(file_id):
    word_dict = {}
    filename = f"./data/source0{file_id}"
    # 读文件 map+combine
    with open(filename, "r") as fr:
        # 一次读一行，以“，”分隔，返回字符串
        line = fr.readline()
        while line:
            for word in line.strip().split(", "):
                word_dict[word] = word_dict.get(word, 0) + 1
            line = fr.readline()
    # shuffle
    word_dict_sorted = dict(sorted(word_dict.items(), key=lambda data: data[0]))
    # 输出map文件
    with open(f"./map/map0{file_id}.txt", "w") as fw:
        for key, val in word_dict_sorted.items():
            fw.write(f"{key}\t{val}\n")
    print(f"finish map0{file_id}")
    # 返回字典
    return word_dict_sorted


# reduce:合并多个map文件的<k,v>
def reducer(word_dicts):
    ans_dict = {}
    for d in word_dicts:
        for k, v in d.items():
            ans_dict[k] = ans_dict.get(k, 0) + v
    return ans_dict


if __name__ == '__main__':
    word_dicts = []
    # 9个线程同时读取数据文件source01~09,并行map+combine
    with futures.ProcessPoolExecutor(9) as pool:
        for word_dict in pool.map(mapper, range(1, 10)):
            word_dicts.append(word_dict)
    print("finish mapping")
    # 开始reduce
    # 分成两部分，并行reduce
    split_dicts = [word_dicts[:5], word_dicts[5:]]
    word_dicts = []
    # 2个线程同时reduce
    with futures.ProcessPoolExecutor(2) as pool:
        for word_dict in pool.map(reducer, split_dicts):
            word_dicts.append(word_dict)
    print("finish reduce1")
    # 合并2个reduce文件
    ans_dict = reducer(word_dicts)
    print("finish reduce2")
    # 写入结果文件
    with open("./result.txt", "w") as fw:
        for k, v in ans_dict.items():
            fw.write(f"{k}\t{v}\n")
