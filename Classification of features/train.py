import random
import os.path as osp

import numpy as np
import paddle
import paddlers as pdrs
from paddlers import transforms as T


# 定义全局变量

# 随机种子
SEED = 77571
# 数据集存放目录
DATA_DIR = '/home/aistudio/data/data77571/dataset/'
# 训练集`file_list`文件路径
TRAIN_FILE_LIST_PATH = '/home/aistudio/data/data77571/dataset/train.txt'
# 验证集`file_list`文件路径
VAL_FILE_LIST_PATH = '/home/aistudio/data/data77571/dataset/val.txt'
# 测试集`file_list`文件路径
TEST_FILE_LIST_PATH = '/home/aistudio/data/data77571/dataset/test.txt'
# 数据集类别信息文件路径
LABEL_LIST_PATH = '/home/aistudio/data/data77571/dataset/labels.txt'
# 实验目录，保存输出的模型权重和结果
EXP_DIR =  '/home/aistudio/exp/'

random.seed(SEED)
np.random.seed(SEED)
paddle.seed(SEED)

# 构建数据集

# 定义训练和验证时使用的数据变换（数据增强、预处理等）
train_transforms = T.Compose([
    # 将影像缩放到256x256大小
    T.Resize(target_size=256),
    # 以50%的概率实施随机水平翻转
    T.RandomHorizontalFlip(prob=0.5),
    # 将数据归一化到[-1,1]
    T.Normalize(
        mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])

eval_transforms = T.Compose([
    T.Resize(target_size=256),
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

# 使用focal loss作为损失函数
model.losses = dict(
    types=[pdrs.models.ppseg.models.FocalLoss()],
    coef=[1.0]
)

# 制定定步长学习率衰减策略
lr_scheduler = paddle.optimizer.lr.StepDecay(
    0.001,
    step_size=8000,
    gamma=0.5
)
# 构造Adam优化器
optimizer = paddle.optimizer.Adam(
    learning_rate=lr_scheduler,
    parameters=model.net.parameters()
)

if __name__ == "__main__":
    # 执行模型训练
    model.train(
        num_epochs=60,
        train_dataset=train_dataset,
        train_batch_size=16,
        eval_dataset=val_dataset,
        optimizer=optimizer,
        save_interval_epochs=3,
        # 每多少次迭代记录一次日志
        log_interval_steps=100,
        save_dir=EXP_DIR,
        # 是否使用early stopping策略，当精度不再改善时提前终止训练
        early_stop=False,
        # 是否启用VisualDL日志功能
        use_vdl=True,
        # 指定从某个检查点继续训练
        resume_checkpoint=None
    )
