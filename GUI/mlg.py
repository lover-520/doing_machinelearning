# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'mlg.ui'
#
# Created by: PyQt5 UI code generator 5.15.0
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_Form(object):
    def setupUi(self, Form):
        Form.setObjectName("Form")
        Form.resize(1246, 745)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(Form.sizePolicy().hasHeightForWidth())
        Form.setSizePolicy(sizePolicy)
        self.gridLayout_2 = QtWidgets.QGridLayout(Form)
        self.gridLayout_2.setObjectName("gridLayout_2")
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setSizeConstraint(QtWidgets.QLayout.SetFixedSize)
        self.horizontalLayout.setContentsMargins(0, -1, -1, -1)
        self.horizontalLayout.setSpacing(250)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.button_knn = QtWidgets.QPushButton(Form)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(1)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.button_knn.sizePolicy().hasHeightForWidth())
        self.button_knn.setSizePolicy(sizePolicy)
        self.button_knn.setLayoutDirection(QtCore.Qt.RightToLeft)
        self.button_knn.setObjectName("button_knn")
        self.horizontalLayout.addWidget(self.button_knn)
        self.button_exit = QtWidgets.QPushButton(Form)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.button_exit.sizePolicy().hasHeightForWidth())
        self.button_exit.setSizePolicy(sizePolicy)
        self.button_exit.setSizeIncrement(QtCore.QSize(10, 10))
        self.button_exit.setLayoutDirection(QtCore.Qt.LeftToRight)
        self.button_exit.setObjectName("button_exit")
        self.horizontalLayout.addWidget(self.button_exit)
        self.gridLayout_2.addLayout(self.horizontalLayout, 0, 0, 1, 1)
        self.widget_knn = QtWidgets.QWidget(Form)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.widget_knn.sizePolicy().hasHeightForWidth())
        self.widget_knn.setSizePolicy(sizePolicy)
        self.widget_knn.setLayoutDirection(QtCore.Qt.LeftToRight)
        self.widget_knn.setAutoFillBackground(True)
        self.widget_knn.setObjectName("widget_knn")
        self.gridLayout = QtWidgets.QGridLayout(self.widget_knn)
        self.gridLayout.setSizeConstraint(QtWidgets.QLayout.SetDefaultConstraint)
        self.gridLayout.setHorizontalSpacing(47)
        self.gridLayout.setObjectName("gridLayout")
        self.verticalLayout_4 = QtWidgets.QVBoxLayout()
        self.verticalLayout_4.setSpacing(47)
        self.verticalLayout_4.setObjectName("verticalLayout_4")
        self.button_read_data_knn = QtWidgets.QPushButton(self.widget_knn)
        self.button_read_data_knn.setObjectName("button_read_data_knn")
        self.verticalLayout_4.addWidget(self.button_read_data_knn)
        self.button_norm_data_knn = QtWidgets.QPushButton(self.widget_knn)
        self.button_norm_data_knn.setObjectName("button_norm_data_knn")
        self.verticalLayout_4.addWidget(self.button_norm_data_knn)
        self.button_class_test_knn = QtWidgets.QPushButton(self.widget_knn)
        self.button_class_test_knn.setObjectName("button_class_test_knn")
        self.verticalLayout_4.addWidget(self.button_class_test_knn)
        self.button_classify_knn = QtWidgets.QPushButton(self.widget_knn)
        self.button_classify_knn.setObjectName("button_classify_knn")
        self.verticalLayout_4.addWidget(self.button_classify_knn)
        self.gridLayout.addLayout(self.verticalLayout_4, 0, 0, 1, 1)
        self.verticalLayout_3 = QtWidgets.QVBoxLayout()
        self.verticalLayout_3.setContentsMargins(20, 0, 20, 50)
        self.verticalLayout_3.setSpacing(7)
        self.verticalLayout_3.setObjectName("verticalLayout_3")
        self.label_ho_ratio_name = QtWidgets.QLabel(self.widget_knn)
        self.label_ho_ratio_name.setAlignment(QtCore.Qt.AlignCenter)
        self.label_ho_ratio_name.setObjectName("label_ho_ratio_name")
        self.verticalLayout_3.addWidget(self.label_ho_ratio_name)
        self.lineEdit_ho_ratio = QtWidgets.QLineEdit(self.widget_knn)
        self.lineEdit_ho_ratio.setAlignment(QtCore.Qt.AlignCenter)
        self.lineEdit_ho_ratio.setObjectName("lineEdit_ho_ratio")
        self.verticalLayout_3.addWidget(self.lineEdit_ho_ratio)
        self.label_error_rate = QtWidgets.QLabel(self.widget_knn)
        self.label_error_rate.setAlignment(QtCore.Qt.AlignCenter)
        self.label_error_rate.setObjectName("label_error_rate")
        self.verticalLayout_3.addWidget(self.label_error_rate)
        self.lineEdit_error_rate = QtWidgets.QLineEdit(self.widget_knn)
        self.lineEdit_error_rate.setAlignment(QtCore.Qt.AlignCenter)
        self.lineEdit_error_rate.setObjectName("lineEdit_error_rate")
        self.verticalLayout_3.addWidget(self.lineEdit_error_rate)
        self.label_features = QtWidgets.QLabel(self.widget_knn)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_features.sizePolicy().hasHeightForWidth())
        self.label_features.setSizePolicy(sizePolicy)
        self.label_features.setLayoutDirection(QtCore.Qt.LeftToRight)
        self.label_features.setAlignment(QtCore.Qt.AlignCenter)
        self.label_features.setObjectName("label_features")
        self.verticalLayout_3.addWidget(self.label_features)
        self.verticalLayout_2 = QtWidgets.QVBoxLayout()
        self.verticalLayout_2.setSpacing(17)
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.lineEdit_feature1 = QtWidgets.QLineEdit(self.widget_knn)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.lineEdit_feature1.sizePolicy().hasHeightForWidth())
        self.lineEdit_feature1.setSizePolicy(sizePolicy)
        self.lineEdit_feature1.setAlignment(QtCore.Qt.AlignCenter)
        self.lineEdit_feature1.setObjectName("lineEdit_feature1")
        self.verticalLayout_2.addWidget(self.lineEdit_feature1)
        self.lineEdit_feature2 = QtWidgets.QLineEdit(self.widget_knn)
        self.lineEdit_feature2.setAlignment(QtCore.Qt.AlignCenter)
        self.lineEdit_feature2.setObjectName("lineEdit_feature2")
        self.verticalLayout_2.addWidget(self.lineEdit_feature2)
        self.lineEdit_feature3 = QtWidgets.QLineEdit(self.widget_knn)
        self.lineEdit_feature3.setAlignment(QtCore.Qt.AlignCenter)
        self.lineEdit_feature3.setObjectName("lineEdit_feature3")
        self.verticalLayout_2.addWidget(self.lineEdit_feature3)
        self.verticalLayout_3.addLayout(self.verticalLayout_2)
        self.label = QtWidgets.QLabel(self.widget_knn)
        self.label.setAlignment(QtCore.Qt.AlignCenter)
        self.label.setObjectName("label")
        self.verticalLayout_3.addWidget(self.label)
        self.lineEdit_classify_result = QtWidgets.QLineEdit(self.widget_knn)
        self.lineEdit_classify_result.setAlignment(QtCore.Qt.AlignCenter)
        self.lineEdit_classify_result.setObjectName("lineEdit_classify_result")
        self.verticalLayout_3.addWidget(self.lineEdit_classify_result)
        self.gridLayout.addLayout(self.verticalLayout_3, 0, 1, 1, 1)
        self.verticalLayout = QtWidgets.QVBoxLayout()
        self.verticalLayout.setObjectName("verticalLayout")
        self.label_data_name = QtWidgets.QLabel(self.widget_knn)
        self.label_data_name.setAlignment(QtCore.Qt.AlignCenter)
        self.label_data_name.setObjectName("label_data_name")
        self.verticalLayout.addWidget(self.label_data_name)
        self.tableview_data = QtWidgets.QTableView(self.widget_knn)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.tableview_data.sizePolicy().hasHeightForWidth())
        self.tableview_data.setSizePolicy(sizePolicy)
        self.tableview_data.setLayoutDirection(QtCore.Qt.LeftToRight)
        self.tableview_data.setAutoFillBackground(True)
        self.tableview_data.setObjectName("tableview_data")
        self.verticalLayout.addWidget(self.tableview_data)
        self.gridLayout.addLayout(self.verticalLayout, 0, 2, 1, 1)
        self.gridLayout.setColumnStretch(0, 1)
        self.gridLayout.setColumnStretch(1, 1)
        self.gridLayout.setColumnStretch(2, 6)
        self.gridLayout_2.addWidget(self.widget_knn, 1, 0, 1, 1)

        self.retranslateUi(Form)
        QtCore.QMetaObject.connectSlotsByName(Form)

    def retranslateUi(self, Form):
        _translate = QtCore.QCoreApplication.translate
        Form.setWindowTitle(_translate("Form", "算法"))
        self.button_knn.setText(_translate("Form", "KNN算法"))
        self.button_exit.setText(_translate("Form", "退出"))
        self.button_read_data_knn.setText(_translate("Form", "读取数据"))
        self.button_norm_data_knn.setText(_translate("Form", "归一化"))
        self.button_class_test_knn.setText(_translate("Form", "效果测试"))
        self.button_classify_knn.setText(_translate("Form", "预测"))
        self.label_ho_ratio_name.setText(_translate("Form", "分割"))
        self.label_error_rate.setText(_translate("Form", "错误率"))
        self.label_features.setText(_translate("Form", "三特征值"))
        self.label.setText(_translate("Form", "预测类别"))
        self.label_data_name.setText(_translate("Form", "数据"))