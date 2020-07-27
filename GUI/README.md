# 文件简介

可视化界面采用Python的PyQt5库进行实现；  
安装了PyQt5和PyQt5-tools等库，利用designer.exe进行页面的设计，然后转为py文件，然后继承这个生成的py文件中的类，添加自己的属性进行实现。（在Pycharm中配置PyQt5的过程略，网上很多）

* mlg.ui：是自己利用designer.exe进行设计的界面；
* mlg.py：是将ui文件转为py文件生成的；
* algorithm_ui.py：继承mlg.py中的类，添加自己的逻辑操作；
* algorithm_knn.py：当时想为每种算法写一个界面，目前还没实现，可不管；
* test.py：测试PyQt5库的文件，可不管