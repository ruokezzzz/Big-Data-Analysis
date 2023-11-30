import pandas as pd

SUPPORT = 0.005
CONF = 0.5


def read_baskets():
    df = pd.read_csv("./Groceries.csv")     # read baskets
    baskets = []   # 每个列表元素是一个basket
    for basket in df["items"]:      # 此时basket为字符串类型，代表一个basket
        baskets.append(set(basket[1:-1].split(",")))
        # split：以“,”为分隔符对basket字符串进行切片，返回切片后的字符串列表
        # set：将字符串列表转换为集合类型，basket从字符串转换为集合
        # append：将该basket集合加入baskets列表
    # print(baskets)
    return baskets


# 文件->C1->L1
def generate_l1(baskets):
    # 用字典记录每个item的频数,获得C1
    C1 = {}
    for basket in baskets:
        for item in basket:
            C1[item] = C1.get(item, 0) + 1
    # 计算C1中每个item的支持度，获得L1，并写入L1.txt
    L1 = C1
    with open("./L1.txt", "w") as fw:
        for item in list(L1.keys()):
            support = L1[item] / len(baskets)    # 计算每个item的支持度
            if support > SUPPORT:
                L1[item] = support
                fw.write(f"{item}: {support}\n")
            else:
                del L1[item]
        fw.write(f"L1 quantities: {len(L1)}\n")
    print(f"1阶频繁项集数量为{len(L1)},保存在L1.txt")
    return L1


# L1->C2->L2
def generate_l2(L1, baskets):
    items = list(L1.keys())  # L1中的所有item
    # 构造C2，记录每个pair的频数
    C2 = {}
    for i in range(0, len(items)):
        for j in range(i+1, len(items)):
            pair = (items[i], items[j])
            for basket in baskets:
                if pair[0] in basket and pair[1] in basket:
                    C2[pair] = C2.get(pair, 0) + 1
    # 构造L2
    L2 = C2
    with open("./L2.txt", "w") as fw:
        for item in list(L2.keys()):
            support = L2[item] / len(baskets)
            if support > SUPPORT:
                L2[item] = support
                fw.write(f"{item}: {support}\n")
            else:
                del L2[item]
        fw.write(f"L2 quantities: {len(L2)}\n")
    print(f"2阶频繁项集数量为{len(L2)},保存在L2.txt")
    return L2


# L2->C3->L3
def generate_l3(L2, baskets):
    # 提取L2中所有item(利用集合的特性去重)
    items = set()
    for pair in list(L2.keys()):
        items.add(pair[0])
        items.add(pair[1])
    items = list(items)
    items.sort()
    # 获取C3，并计算频数
    C3 = {}
    for i in range(0, len(items)):
        for j in range(i+1, len(items)):
            for k in range(j+1,  len(items)):
                tup = (items[i], items[j], items[k])
                for basket in baskets:
                    if tup[0] in basket and tup[1] in basket and tup[2] in basket:
                        C3[tup] = C3.get(tup, 0)+1
    L3 = C3
    with open("./L3.txt", "w") as fw:
        for tup in list(L3.keys()):
            support = L3[tup] / len(baskets)
            if support > SUPPORT:
                L3[tup] = support
                fw.write(f"{tup}: {support}\n")
            else:
                del L3[tup]
        fw.write(f"L3 quantities: {len(L3)}\n")
    print(f"3阶频繁项集数量为{len(L3)},保存在L3.txt")
    return L3


# L1,L2->Rules2
def assorules_2(L1, L2):
    items = list(L2.keys())
    rules2_num = 0
    with open("./Rules2.txt", "w") as fw:
        for pair in items:
            conf = L2[pair] / L1[pair[0]]
            if conf > CONF:
                fw.write(f"{pair[0]}->{pair[1]}: {conf}\n")
                rules2_num += 1

            conf = L2[pair] / L1[pair[1]]
            if conf > CONF:
                fw.write(f"{pair[1]}->{pair[0]}: {conf}\n")
                rules2_num += 1
        fw.write(f"2-Rules quantities: {rules2_num}\n")
    print(f"2阶关联规则数量为{rules2_num},保存在Rules2.txt")


# L1,L2,L3->Rules3
def assorules_3(L1, L2, L3):
    rules3_num = 0
    with open("./Rules3.txt", "w") as fw:
        for tup in list(L3.keys()):
            conf = L3[tup] / L1[tup[0]]
            if conf >= CONF:
                fw.write(f"{tup[0]}->({tup[1]},{tup[2]}): {conf}\n")
                rules3_num += 1
            conf = L3[tup] / L1[tup[1]]
            if conf >= CONF:
                fw.write(f"{tup[1]}->({tup[0]},{tup[2]}): {conf}\n")
                rules3_num += 1
            conf = L3[tup] / L1[tup[2]]
            if conf >= CONF:
                fw.write(f"{tup[2]}->({tup[0]},{tup[1]}): {conf}\n")
                rules3_num += 1

            if (tup[0], tup[1]) in L2:
                conf = L3[tup] / L2[(tup[0], tup[1])]
                if conf >= CONF:
                    fw.write(f"({tup[0]},{tup[1]})->{tup[2]}: {conf}\n")
                    rules3_num += 1
            elif (tup[1], tup[0]) in L2:
                conf = L3[tup] / L2[(tup[1], tup[0])]
                if conf >= CONF:
                    fw.write(f"({tup[0]},{tup[1]})->{tup[2]}: {conf}\n")
                    rules3_num += 1
            else:
                print((tup[0], tup[1]), "not found")

            if (tup[0], tup[2]) in L2:
                conf = L3[tup] / L2[(tup[0], tup[2])]
                if conf >= CONF:
                    fw.write(f"({tup[0]},{tup[2]})->{tup[1]}: {conf}\n")
                    rules3_num += 1
            elif (tup[2], tup[0]) in L2:
                conf = L3[tup] / L2[(tup[2], tup[0])]
                if conf >= CONF:
                    fw.write(f"({tup[0]},{tup[2]})->{tup[1]}: {conf}\n")
                    rules3_num += 1
            else:
                print((tup[0], tup[2]), "not found")

            if (tup[1], tup[2]) in L2:
                conf = L3[tup] / L2[(tup[1], tup[2])]
                if conf >= CONF:
                    fw.write(f"({tup[1]},{tup[2]})->{tup[0]}: {conf}\n")
                    rules3_num += 1
            elif (tup[2], tup[1]) in L2:
                conf = L3[tup] / L2[(tup[2], tup[1])]
                if conf >= CONF:
                    fw.write(f"({tup[1]},{tup[2]})->{tup[0]}: {conf}\n")
                    rules3_num += 1
            else:
                print((tup[1], tup[2]), "not found")

        fw.write(f"3-Rules quantities: {rules3_num}\n")
    print(f"3阶关联规则数量为{rules3_num},保存在Rules3.txt")


if __name__ == '__main__':
    baskets = read_baskets()
    L1 = generate_l1(baskets)
    L2 = generate_l2(L1, baskets)
    assorules_2(L1, L2)
    L3 = generate_l3(L2, baskets)
    assorules_3(L1, L2, L3)