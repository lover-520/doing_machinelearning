# -*- coding: utf-8 -*-
"""
@function: 存储所有关于knn算法的函数
"""

from PyQt5.QtWidgets import QWidget, QMessageBox, QTableView
from PyQt5.QtCore import *
from PyQt5.QtGui import QStandardItemModel, QStandardItem


# def prompt_message():
#     QMessageBox.information(QWidget(), '算法介绍', '这是KNN算法的Demo', QMessageBox.Yes)
#
#
# def read_data_fun():
#     print('读取数据')
#     model = QStandardItemModel(4, 4)
#     model.setHorizontalHeaderLabels(['年飞行里程数', '游戏百分比', '冰淇淋数', '类别'])
#     for row in range(4):
#         for column in range(4):
#             i = QStandardItem("row %s, column %s" % (row, column))
#             model.setItem(row, column, i)
#     return model
