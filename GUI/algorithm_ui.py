# -*- coding: utf-8 -*-
from PyQt5.QtGui import QStandardItemModel, QStandardItem
from PyQt5.QtWidgets import QWidget, QMessageBox, QHeaderView, QLineEdit
from PyQt5.QtCore import Qt

import numpy as np

from GUI import mlg
from KNN import knn


class AlgorithmUI(QWidget, mlg.Ui_Form):
    def __init__(self, parent=None):
        super(AlgorithmUI, self).__init__(parent)
        self.setupUi(self)
        '''
            标准模块
        '''
        # 添加标准的结束按钮
        self.button_exit.clicked.connect(self.close)

        '''
        knn算法模块
        '''
        # 添加KNN算法按钮信号和槽

        # 设置文本框的显示字幕，应该初始化整个界面时就应该显示
        self.lineEdit_ho_ratio.setPlaceholderText('分割率')
        self.lineEdit_ho_ratio.setEchoMode(QLineEdit.Normal)
        self.lineEdit_error_rate.setPlaceholderText('错误率')
        self.lineEdit_error_rate.setEchoMode(QLineEdit.Normal)
        self.lineEdit_error_rate.setReadOnly(True)
        self.lineEdit_feature1.setPlaceholderText('年飞行里程数')
        self.lineEdit_feature1.setEchoMode(QLineEdit.Normal)
        self.lineEdit_feature2.setPlaceholderText('游戏百分比')
        self.lineEdit_feature2.setEchoMode(QLineEdit.Normal)
        self.lineEdit_feature3.setPlaceholderText('冰淇淋数')
        self.lineEdit_feature3.setEchoMode(QLineEdit.Normal)
        self.lineEdit_classify_result.setPlaceholderText('预测结果')
        self.lineEdit_classify_result.setEchoMode(QLineEdit.Normal)
        self.lineEdit_classify_result.setReadOnly(True)

        self.button_knn.clicked.connect(self.prompt_message_knn)  # 给knn算法按钮添加显示信息
        self.data_mat, self.class_labels = [], []
        self.button_read_data_knn.clicked.connect(self.read_data_fun_knn)  # knn算法读取数据的按钮
        self.button_norm_data_knn.clicked.connect(self.norm_data_fun_knn)  # 归一化数据按钮
        self.button_class_test_knn.clicked.connect(self.class_test_fun_knn)  # 测试按钮
        self.button_classify_knn.clicked.connect(self.classify_person_fun_knn)  # 预测按钮

    def prompt_message_knn(self):
        QMessageBox.information(QWidget(), '算法介绍', '这是KNN算法的Demo', QMessageBox.Yes)

    def show_table_view(self, data_mat, class_labels):
        model = QStandardItemModel(len(class_labels), 4)
        model.setHorizontalHeaderLabels(['年飞行里程数', '游戏百分比', '冰淇淋数', '类别'])
        for row in range(len(class_labels)):
            for column in range(4):
                if column != 3:
                    i = QStandardItem("%s" % data_mat[row][column])
                else:
                    i = QStandardItem(class_labels[row])
                i.setTextAlignment(Qt.AlignHCenter | Qt.AlignVCenter)
                model.setItem(row, column, i)
        self.tableview_data.setModel(model)
        self.tableview_data.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)

    def read_data_fun_knn(self):
        self.data_mat, self.class_labels = knn.file2matrix('./KNN/datingTestSet2.txt')
        self.show_table_view(data_mat=self.data_mat, class_labels=self.class_labels)

    def norm_data_fun_knn(self):
        # min_value = self.data_mat.min(0)  # 取每一列的最小值
        # max_value = self.data_mat.max(0)  # 同上
        # num_ranges = max_value - min_value
        # m = self.data_mat.shape[0]
        # norm_data = self.data_mat - np.tile(min_value, (m, 1))
        # norm_data = norm_data / np.tile(num_ranges, (m, 1))
        # self.data_mat = norm_data
        self.data_mat, num_ranges, min_value = knn.auto_norm(self.data_mat)
        self.show_table_view(data_mat=self.data_mat, class_labels=self.class_labels)

    def class_test_fun_knn(self):
        s = knn.class_test(self.data_mat, self.class_labels, float(self.lineEdit_ho_ratio.text()))
        self.lineEdit_error_rate.setText(str(s))

    def classify_person_fun_knn(self):
        flying_miles = float(self.lineEdit_feature1.text())
        playing_games_per = float(self.lineEdit_feature2.text())
        icecream_consume = float(self.lineEdit_feature3.text())
        classify_result_tmp = knn.classify_person(self.data_mat, self.class_labels, flying_miles, playing_games_per, icecream_consume)
        self.lineEdit_classify_result.setText(classify_result_tmp)
