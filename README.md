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
*  验证<br>
```python
python infer.py
```
* 参数介绍
 * DATA_DIR 数据集路径<br>
 * out_dir 结果保存路径<br>
# 4.模型导出
* 模型导出与部署[点击此处](https://github.com/PaddleCV-SIG/PaddleRS/tree/develop/deploy/export"模型导出与部署")
