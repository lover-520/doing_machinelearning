from numpy import array, tile, zeros, shape
import operator
import matplotlib.pyplot as plt
import os

'''
@function: 随便创建一组简单的数组进行演示
'''


def create_dataset():
    group = array([[1.0, 1.1], [1.0, 1.0], [0.0, 0.0], [0.0, 0.1]])
    labels = ['A', 'A', 'B', 'B']
    return group, labels


'''
@function: knn分类器实现
@params:
inx: 需进行分类的样本
dataset: 提供的数据集
labels: 提供的数据集的标签
k: 取样本的前k个
'''


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
    # 字典中的items函数返回包含所有键值对的元组，返回值是一个列表
    sorted_class_count = sorted(class_count.items(),
                                key=operator.itemgetter(1), reverse=True)
    return sorted_class_count[0][0]


'''
@function: 将文件内容读取为矩阵形式
文件内容包括：
每年获得的飞行常客里程数
玩视频游戏所消耗游戏百分比
每周消费的冰淇淋公升数
喜欢程度
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


'''
@function: 归一化数据
'''


def auto_norm(data):
    min_value = data.min(0)  # 取每一列的最小值
    max_value = data.max(0)  # 同上
    num_ranges = max_value - min_value
    norm_data = zeros(shape(data))
    m = data.shape[0]
    norm_data = data - tile(min_value, (m, 1))
    norm_data = norm_data / tile(num_ranges, (m, 1))
    return norm_data, num_ranges, min_value


'''
@function: 测试分类器
'''


def class_test(data, labels, ho_ratio):
    norm_data, num_ranges, min_value = auto_norm(data)
    m = norm_data.shape[0]
    num_tests = int(m * ho_ratio)  # 用来作测试的样本数
    error_count = 0.0
    for i in range(num_tests):
        classifier_result = knn(norm_data[i, :],
                                norm_data[num_tests:m, :],
                                labels[num_tests:m], 3)
        # print('the classifier came back with: %s, the real answer is : %s'
        #       % (classifier_result, labels[i]))
        if classifier_result is not labels[i]:
            error_count += 1.0
    # print('the total error rate is: %f' % (error_count / float(num_tests)))
    return error_count / float(num_tests)


'''
@function: 预测
'''


def classify_person(data, labels, flying_miles, playing_games_per, icecream_consume):
    # flying_miles = float(input("每年获得的飞行常客里程数: "))
    # playing_games_per = float(input("玩视频游戏所消耗游戏百分比: "))
    # icecream_consume = float(input("每周消费的冰淇淋公升数: "))
    norm_data, num_ranges, min_value = auto_norm(data)
    person = array([flying_miles, playing_games_per, icecream_consume])
    classifier_result = knn((person - min_value) /
                            num_ranges, norm_data, labels, 3)
    # print('您对这位先生的喜好程度大致为：', classifier_result)
    return classifier_result


def draw_data(data, labels):
    plt.rcParams['font.sans-serif'] = ['Simhei']
    plt.rcParams['axes.unicode_minus'] = False
    fig = plt.figure()
    ax = fig.add_subplot(111)
    # ax.scatter(data_mat[:, 0], data_mat[:, 1],
    #            10.0 * array(class_labels), 10.0 * array(class_labels))
    type1_x, type1_y = [], []
    type2_x, type2_y = [], []
    type3_x, type3_y = [], []
    for i in range(len(labels)):
        if labels[i] is 1:
            type1_x.append(data[i][0])
            type1_y.append(data[i][1])
        if labels[i] is 2:
            type2_x.append(data[i][0])
            type2_y.append(data[i][1])
        if labels[i] is 3:
            type3_x.append(data[i][0])
            type3_y.append(data[i][1])
    type1 = ax.scatter(type1_x, type1_y, s=10, c='r')
    type2 = ax.scatter(type2_x, type2_y, s=10, c='g')
    type3 = ax.scatter(type3_x, type3_y, s=10, c='b')
    plt.legend((type1, type2, type3),
               ('didntLike', 'smallDoses', 'largeDoses'))
    plt.show()


'''
@function: 将图片转化为向量格式
'''


def img2vector(filename):
    returnvect = zeros((1, 1024))
    fr = open(filename)
    for i in range(32):
        linestr = fr.readline()
        for j in range(32):
            returnvect[0, 32*i+j] = int(linestr[j])
    return returnvect


'''
@function: 手写数字识别系统的测试代码
'''


def handwriting_class_test():
	hwlabels = []
	trainingfile_list = os.listdir('digits/trainingDigits/')
	m = len(trainingfile_list)  # 表示里面总共有多少个文件
	training_mat = zeros((m, 1024))
	for i in range(m):
		filename_str = trainingfile_list[i]  # 得到文件名
		filename_str_pre = filename_str.split('.')[0]  # 得到前缀名
		class_num = int(filename_str_pre.split('_')[0])
		hwlabels.append(class_num)
		training_mat[i, :] = img2vector('digits/trainingDigits/%s' %filename_str)
	testfile_list = os.listdir('digits/testDigits/')
	error_count = 0.0
	mTest = len(testfile_list)
	for i in range(mTest):
		filename_str = testfile_list[i]
		filename_str_pre = filename_str.split('.')[0]  # 得到前缀名
		class_num = int(filename_str_pre.split('_')[0])
		vectest = img2vector('digits/testDigits/%s' %filename_str)
		class_result = knn(vectest, training_mat, hwlabels, 3)
		# print('the classifier came back with: %d, the real answer is %d' 
		# 	% (class_result, class_num))
		if class_result != class_num:
			error_count += 1.0
			print('the classifier came back with: %d, the real answer is %d' 
				% (class_result, class_num))
	print('the total number of errors is: %d' % error_count)
	print('the total error rate is: %f' % (error_count/float(mTest)))


if __name__ == '__main__':
    # groups, labels = create_dataset()
    # result = knn([0, 0], groups, labels, 3)
    # print(result)
    # data, labels = file2matrix('datingTestSet2.txt')
    # draw_data(data, labels)
    # print(data)
    # class_test(data, labels, 0.10)
    # print(classify_person(data, labels, 10000.0, 10.0, 0.06))
    # testvect = img2vector('digits/testDigits/0_0.txt')
    # print(testvect[0, 0:31])
    # print(testvect[0, 32:63])
    handwriting_class_test()
