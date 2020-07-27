# -*- coding: utf-8 -*-

from numpy import zeros, shape, tile

'''
@function: 这个文件主要用于检验同一数据多次归一化操作之后数据还会发生变化不

@result: 结果证明对于同一数据而言，只有第一次归一化有用，后面的归一化操作不会改变数据
'''


def file2matrix(filename):
    fr = open(filename)
    lines = fr.readlines()
    num_lines = len(lines)
    data = zeros((num_lines, 3))  # 数据矩阵
    labels = []  # 标签
    str_labels = ["didn't like", "small doses", "large doses"]
    index = 0
    for line in lines:
        line = line.strip()  # 去掉行的首尾字符，默认为空格
        list_line = line.split('\t')  # 以\t符号分割得到四个值
        data[index, :] = list_line[0:3]  # 取前三个值放入数据矩阵中
        labels.append(str_labels[int(list_line[-1])-1])  # 将最后一个标签值放去标签列表中
        index += 1
    return data, labels  # 返回数据矩阵和标签值


def auto_norm(data):
    min_value = data.min(0)  # 取每一列的最小值
    max_value = data.max(0)  # 同上
    num_ranges = max_value - min_value
    norm_data = zeros(shape(data))
    m = data.shape[0]
    norm_data = data - tile(min_value, (m, 1))
    norm_data = norm_data / tile(num_ranges, (m, 1))
    return norm_data, num_ranges, min_value


if __name__ == '__main__':
    data, labels = file2matrix('F:\MyProject\doing_machinelearning\KNN\datingTestSet2.txt')
    # for i in range(3):
    #     data, _, _ = auto_norm(data)
    #     print('第 %d 次结果：' % i)
    #     print(data[0:3])
    norm_data1, _, _ = auto_norm(data)
    print(norm_data1[0:3])
    norm_data2, _, _ = auto_norm(norm_data1)
    print(norm_data2[0:3])
    norm_data3, _, _ = auto_norm(norm_data2)
    print(norm_data3[0:3])
