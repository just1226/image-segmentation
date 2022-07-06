# image-segmentation
# 1.环境依赖
* PaddlePaddle >= 2.2.0<br>
* PaddleRS安装<br>
```python
git clone https://github.com/PaddleCV-SIG/PaddleRS
cd PaddleRS
git checkout develop
pip install -r requirements.txt
python setup.py install
```
* python >= 3.7 <br>
* 安装相应库
```python
cd image-segementation
pip install -r requirements.txt
```
# 2.数据集下载
* 变化检测数据集下载地址<br>
https://aistudio.baidu.com/aistudio/datasetdetail/134796 <rb>
```python
# 解压数据集
# 该操作涉及大量文件IO，可能需要一些时间
!unzip -o -d /home/aistudio/data/data134796/dataset /home/aistudio/data/data134796/train_data.zip > /dev/null
!unzip -o -d /home/aistudio/data/data134796/dataset /home/aistudio/data/data134796/test_data.zip > /dev/null
```
* 地物分类数据集下载地址<br>
https://aistudio.baidu.com/aistudio/datasetdetail/77571 <br>
```python
# 解压数据集
!unzip -oq -d /home/aistudio/data/data77571/dataset/ /home/aistudio/data/data77571/train_and_label.zip
```
* 目标检测数据集下载地址 <br>
https://aistudio.baidu.com/aistudio/datasetdetail/52980 <br>
```python
# 解压数据集
!unzip -oq -d /home/aistudio/data/data52980/rsod/ /home/aistudio/data/data52980/RSOD-Dataset.zip
!unzip -oq -d /home/aistudio/data/data52980/rsod/ /home/aistudio/data/data52980/rsod/RSOD-Dataset/playground.zip
```
* 目标提取数据集下载地址 <br>
https://aistudio.baidu.com/aistudio/datasetdetail/56961 <br>
```python
# 解压数据集
!unzip -oq -d /home/aistudio/data/data56961/massroad /home/aistudio/data/data56961/mass_road.zip
```
# 3.模型训练与测试
* 以变化检测为例<br>
*   训练<br>
```python
cd image-segmentation/Change detection/
python dataset.py
python BIT-trains.py
```
* 参数介绍<br>
  *  DATA_DIR 数据集路径<br>
  *  EXP_DIR 模型权重和日志存放路径<br>
  *  BEST_CKP_PATH 最佳模型路径<br>
*  测试<br>
```python
python infer.py
```
* 参数介绍
 * DATA_DIR 数据集路径<br>
 * out_dir 结果保存路径<br>
* 也可以下载我们训练好权重文件，分别放到对应的文件夹下即可。[百度云链接](https://pan.baidu.com/s/1YDVUJiu-jus-Ur17eTPgYA),提取码 bda4
# 4.模型导出
训练完成后可以通过下面的方式导出模型<br>
* 模型导出与部署[点击此处](https://github.com/PaddleCV-SIG/PaddleRS/tree/develop/deploy/export)
* 我们提供模型导出权重文件下载 [百度云链接](https://pan.baidu.com/s/10OurgBOMNVeun8DTRbFoIg),提取码 bda4 <br>
`***将tq_export.zip解压后改为export放到指定路径下***`
# 5.web运行
* 将本项目克隆到本地后，导出训练好的模型权重文件或者下载我们提供的模型导出权重文件放到指定文件夹下<br>
```python
python app.py
```
打开网页输入 `http://127.0.0.1:5000/`即可运行web
# 6.代码结构与详细说明
```python
image-segmentation
├── Change detection
      ├── BIT-trains.py       # 变化检测模型训练
      ├── dataset.py          # 数据处理
      ├── infer.py            # 变化检测模型推理
      ├── models.py           # 变化检测模型
├── Classification of features
      ├── dataset.py          # 数据处理
      ├── infer.py            # 地物分类模型推理
      ├── trains.py           # 地物分类模型训练
├── Object detection
      ├── dataset.py          # 数据处理
      ├── infer.py            # 目标检测模型推理
      ├── trains.py           # 目标检测模型训练
├── Target extraction
      ├── dataset.py          # 数据处理
      ├── infer.py            # 目标提取模型推理
      ├── trains.py           # 目标提取模型训练
├── web                       # web源码
      ├── BIT                 
           ├── static_model   # 变化检测导出模型权重
           ├── test.py        # 变换检测导出模型
      ├── classify            
           ├── export         # 地物分类导出模型权重
           ├── cls_test.py    # 地物分类导出模型
      ├── PaddleRS-develop    # paddlers
      ├── PPYOLO              
           ├── PPYOLO_model   # 目标检测导出模型权重
           ├── main.py        # 目标检测导出模型
      ├── saved_files         # 测试图片
           ├── bit            # 变化检测测试图片
           ├── cls            # 地物分类测试图片
           ├── odt            # 目标检测测试图片
           ├── tq             # 目标提取测试图片
      ├── saved_imgs          # 结果保存
           ├── bit            # 变化检测结果
           ├── cls            # 地物分类结果
           ├── odt            # 目标检测结果
           ├── tq             # 目标提取结果
      ├── static              # 网页设计
      ├── templates           # 网页设计
      ├── tq_best              
           ├── export         # 目标提取导出模型权重
           ├── main.py        # 目标提取导出模型
      ├── app.py              # web启动
├── requirement.txt      # 相应库
```


