# 导入需要用到的库

import random
import os.path as osp

import numpy as np
import paddle
import paddlers as pdrs
from paddlers import transforms as T

# 定义全局变量

# 随机种子
SEED = 56961
# 数据集存放目录
DATA_DIR = '/home/aistudio/data/data56961/massroad/'
# 训练集`file_list`文件路径
TRAIN_FILE_LIST_PATH = '/home/aistudio/data/data56961/massroad/train.txt'
# 验证集`file_list`文件路径
VAL_FILE_LIST_PATH = '/home/aistudio/data/data56961/massroad/val.txt'
# 测试集`file_list`文件路径
TEST_FILE_LIST_PATH = '/home/aistudio/data/data56961/massroad/test.txt'
# 数据集类别信息文件路径
LABEL_LIST_PATH = '/home/aistudio/data/data56961/massroad/labels.txt'
# 实验目录，保存输出的模型权重和结果
EXP_DIR =  '/home/aistudio/exp/'

# 固定随机种子，尽可能使实验结果可复现

random.seed(SEED)
np.random.seed(SEED)
paddle.seed(SEED)

# 构建数据集

# 定义训练和验证时使用的数据变换（数据增强、预处理等）
train_transforms = T.Compose([
    # 随机裁剪
    T.RandomCrop(crop_size=512),
    # 以50%的概率实施随机水平翻转
    T.RandomHorizontalFlip(prob=0.5),
    # 以50%的概率实施随机垂直翻转
    T.RandomVerticalFlip(prob=0.5),
    # 将数据归一化到[-1,1]
    T.Normalize(
        mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])

eval_transforms = T.Compose([
    T.Resize(target_size=1488),
    # 验证阶段与训练阶段的数据归一化方式必须相同
    T.Normalize(
        mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])

# 分别构建训练和验证所用的数据集
train_dataset = pdrs.datasets.SegDataset(
    data_dir=DATA_DIR,
    file_list=TRAIN_FILE_LIST_PATH,
    label_list=LABEL_LIST_PATH,
    transforms=train_transforms,
    num_workers=4,
    shuffle=True
)

val_dataset = pdrs.datasets.SegDataset(
    data_dir=DATA_DIR,
    file_list=VAL_FILE_LIST_PATH,
    label_list=LABEL_LIST_PATH,
    transforms=eval_transforms,
    num_workers=0,
    shuffle=False
)

# 构建DeepLab V3+模型，使用ResNet-50作为backbone
model = pdrs.tasks.DeepLabV3P(
    input_channel=3,
    num_classes=len(train_dataset.labels),
    backbone='ResNet50_vd'
)
model.net_initialize(
    pretrain_weights='CITYSCAPES',
    save_dir=osp.join(EXP_DIR, 'pretrain'),
    resume_checkpoint=None,
    is_backbone_weights=False
)
# 构建优化器
optimizer = paddle.optimizer.Adam(
    learning_rate=0.001,
    parameters=model.net.parameters()
)

if __name__ == "__main__":
    # 执行模型训练
    model.train(
        num_epochs=100,
        train_dataset=train_dataset,
        train_batch_size=8,
        eval_dataset=val_dataset,
        optimizer=optimizer,
        save_interval_epochs=1,
        # 每多少次迭代记录一次日志
        log_interval_steps=30,
        save_dir=EXP_DIR,
        # 是否使用early stopping策略，当精度不再改善时提前终止训练
        early_stop=False,
        # 是否启用VisualDL日志功能
        use_vdl=True,
        # 指定从某个检查点继续训练
        resume_checkpoint=None
    )
