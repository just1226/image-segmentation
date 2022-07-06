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
