from numpy import array, tile, zeros, shape
import operator
import matplotlib.pyplot as plt


def create_dataset():
    group = array([[1.0, 1.1], [1.0, 1.0], [0.0, 0.0], [0.0, 0.1]])
    labels = ['A', 'A', 'B', 'B']
    return group, labels


def knn(inx, dataset, labels, k):
    dataset_size = dataset.shape[0]  # 表示里面有多少个点
    diff_mat = tile(inx, (dataset_size, 1)) - dataset  # 在行列上重复该点并相减
    distances_mat = (diff_mat ** 2).sum(axis=1)  # 得到每个点的距离只
    distances_mat = distances_mat ** 0.5
    sorted_dist_indices = distances_mat.argsort()  # 将距离值排序，返回索引值
    class_count = {}  # 得到前k个点中每个类别出现的次数
    for i in range(k):
        vote_label = labels[sorted_dist_indices[i]]
        # 字典中的get函数返回该键对应的值， 默认值为第二个参数
        class_count[vote_label] = class_count.get(vote_label, 0) + 1
    # 字典中的items函数返回包含所有键值对的元组
    sorted_class_count = sorted(class_count.items(),
                                key=operator.itemgetter(1), reverse=True)
    return sorted_class_count[0][0]


'''
@function: 将文件内容读取为矩阵形式
'''


def file2matrix(filename):
    fr = open(filename)
    lines = fr.readlines()
    num_lines = len(lines)
    data_mat = zeros((num_lines, 3))  # 数据矩阵
    class_labels = []  # 标签
    index = 0
    for line in lines:
        line = line.strip()  # 去掉行的首尾字符，默认为空格
        list_line = line.split('\t')  # 以\t符号分割得到四个值
        data_mat[index, :] = list_line[0:3]  # 取前三个值放入数据矩阵中
        class_labels.append(int(list_line[-1]))  # 将最后一个标签值放去标签列表中
        index += 1
    return data_mat, class_labels  # 返回数据矩阵和标签值


'''
@function: 归一化数据
'''


def autonorm(data_mat):
    min_value = data_mat.min(0)  # 取每一列的最小值
    max_value = data_mat.max(0)  # 同上
    num_ranges = max_value - min_value
    norm_data_mat = zeros(shape(data_mat))
    m = data_mat.shape[0]
    norm_data_mat = data_mat - tile(min_value, (m, 1))
    norm_data_mat = norm_data_mat / tile(num_ranges, (m, 1))
    return norm_data_mat, num_ranges, min_value


'''
@function: 测试分类器
'''


def classtest(data_mat, class_labels):
    horatio = 0.60
    norm_data_mat, num_ranges, min_value = autonorm(data_mat)
    m = norm_data_mat.shape[0]
    num_vecs = int(m * horatio)
    error_count = 0.0
    for i in range(num_vecs):
        classifier_result = knn(norm_data_mat[i, :],
                                norm_data_mat[num_vecs:m, :],
                                class_labels[num_vecs:m], 3)
        print('the classifier came back with: %d, the real answer is : %d'
              % (classifier_result, class_labels[i]))
        if classifier_result is not class_labels[i]:
            error_count += 1.0
    print('the total error rate is: %f' % (error_count / float(num_vecs)))


def draw_data(data_mat, class_labels):
    plt.rcParams['font.sans-serif'] = ['Simhei']
    plt.rcParams['axes.unicode_minus'] = False
    fig = plt.figure()
    ax = fig.add_subplot(111)
    # ax.scatter(data_mat[:, 0], data_mat[:, 1],
    #            10.0 * array(class_labels), 10.0 * array(class_labels))
    type1_x, type1_y = [], []
    type2_x, type2_y = [], []
    type3_x, type3_y = [], []
    for i in range(len(class_labels)):
        if class_labels[i] is 1:
            type1_x.append(data_mat[i][0])
            type1_y.append(data_mat[i][1])
        if class_labels[i] is 2:
            type2_x.append(data_mat[i][0])
            type2_y.append(data_mat[i][1])
        if class_labels[i] is 3:
            type3_x.append(data_mat[i][0])
            type3_y.append(data_mat[i][1])
    type1 = ax.scatter(type1_x, type1_y, s=10, c='r')
    type2 = ax.scatter(type2_x, type2_y, s=10, c='g')
    type3 = ax.scatter(type3_x, type3_y, s=10, c='b')
    plt.legend((type1, type2, type3), ('didntLike', 'smallDoses', 'largeDoses'))
    plt.show()


if __name__ == '__main__':
    # groups, labels = create_dataset()
    # result = knn([0, 0], groups, labels, 3)
    # print(result)
    data_mat, class_labels = file2matrix('datingTestSet2.txt')
    # draw_data(data_mat, class_labels)
    # print(data_mat)
    classtest(data_mat, class_labels)
