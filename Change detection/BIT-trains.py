# 执行此脚本前，请确认已正确安装PaddleRS库
import sys
sys.path.append('/image-segmentation/PaddleRS')
import paddle
import os
import argparse
import paddlers as pdrs
from paddlers import transforms as T

#定义全局变量
def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', '-m', type=str, default=None, help='model directory path')
    parser.add_argument('--out_dir', '-s', type=str, default=None, help='path to save inference model')
    parser.add_argument('-lr', type=float, default=0.0005, help='lr')
    parser.add_argument('--num_epoch', type=int, default=200, help='epoch number')
    parser.add_argument('--train_batch_size', type=int, default=8, help='batch size')
    parser.add_argument('--test_batch_size', type=int, default=8, help='batch size')
    parser.add_argument('--crop_size', type=int, default=(640,640), help='H,W')
    parser.add_argument('--stride', type=int, default=64, help='The sliding window step used for model inference')
    parser.add_argument('--original_size', type=int, default=(1024,1024), help='image size')
    parser.add_argument('--save_epoch', type=int, default=5, help='save epoch')
    parser.add_argument('--num_workers', type=int, default=4, help='The number of processes used to load the data')
    parser.add_argument('--decay_step', type=int, default=1000, help='Learning rate decay step size')

    return parser
if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    # 随机种子
    SEED = 1919810
    
    DATA_DIR = args.data_dir
    EXP_DIR = args.out_dir
    NUM_EPOCHS = args.num_epoch
    SAVE_INTERVAL_EPOCHS = args.save_epoch
    LR = args.lr
    DECAY_STEP = args.decay_step
    TRAIN_BATCH_SIZE = args.train_batch_size
    INFER_BATCH_SIZE = args.test_batch_size
    NUM_WORKERS = args.num_workers
    CROP_SIZE = args.crop_size
    STRIDE = args.stride
    ORIGINAL_SIZE = args.original_size
    # 保存最佳模型的路径
    BEST_CKP_PATH = osp.join(EXP_DIR, 'best_model', 'model.pdparams')
    
    # 调用PaddleRS API一键构建模型
    model = pdrs.tasks.BIT(
        # 模型输出类别数
        num_classes=2,
        # 是否使用混合损失函数，默认使用交叉熵损失函数训练
        use_mixed_loss=False,
        # 模型输入通道数
        in_channels=3,
        # 模型使用的骨干网络，支持'resnet18'或'resnet34'
        backbone='resnet34',
        # 骨干网络中的resnet stage数量
        n_stages=4,
        # 是否使用tokenizer获取语义token
        use_tokenizer=True,
        # token的长度
        token_len=4,
        # 若不使用tokenizer，则使用池化方式获取token。此参数设置池化模式，有'max'和'avg'两种选项，分别对应最大池化与平均池化
        pool_mode='max',
        # 池化操作输出特征图的宽和高（池化方式得到的token的长度为pool_size的平方）
        pool_size=2,
        # 是否在Transformer编码器中加入位置编码（positional embedding）
        enc_with_pos=True,
        # Transformer编码器使用的注意力模块（attention block）个数
        enc_depth=1,
        # Transformer编码器中每个注意力头的嵌入维度（embedding dimension）
        enc_head_dim=64,
        # Transformer解码器使用的注意力模块个数
        dec_depth=4,
        # Transformer解码器中每个注意力头的嵌入维度
        dec_head_dim=8
    )
    # 构建需要使用的数据变换（数据增强、预处理）
    # 使用Compose组合多种变换方式。Compose中包含的变换将按顺序串行执行
    train_transforms = T.Compose([
        # 随机裁剪
        T.RandomCrop(
            # 裁剪区域将被缩放到此大小
            crop_size=CROP_SIZE,
            # 将裁剪区域的横纵比固定为1
            aspect_ratio=[1.0, 1.0],
            # 裁剪区域相对原始影像长宽比例在一定范围内变动，最小不低于原始长宽的1/5
            scaling=[0.2, 1.0]
        ),
        # 以50%的概率实施随机水平翻转
        T.RandomHorizontalFlip(prob=0.5),
        # 以50%的概率实施随机垂直翻转
        T.RandomVerticalFlip(prob=0.5),
        # 数据归一化到[-1,1]
        T.Normalize(
            mean=[0.5, 0.5, 0.5],
            std=[0.5, 0.5, 0.5]
        )
    ])
    eval_transforms = T.Compose([
        # 在验证阶段，输入原始尺寸影像，对输入影像仅进行归一化处理
        # 验证阶段与训练阶段的数据归一化方式必须相同
        T.Normalize(
            mean=[0.5, 0.5, 0.5],
            std=[0.5, 0.5, 0.5]
        )
    ])

    # 实例化数据集
    train_dataset = pdrs.datasets.CDDataset(
        data_dir=DATA_DIR,
        file_list=osp.join(DATA_DIR, 'train.txt'),
        label_list=None,
        transforms=train_transforms,
        num_workers=NUM_WORKERS,
        shuffle=True,
        binarize_labels=True
    )
    eval_dataset = pdrs.datasets.CDDataset(
        data_dir=DATA_DIR,
        file_list=osp.join(DATA_DIR, 'val.txt'),
        label_list=None,
        transforms=eval_transforms,
        num_workers=0,
        shuffle=False,
        binarize_labels=True
    )
    if not osp.exists(EXP_DIR):
        os.makedirs(EXP_DIR)
    # 构建学习率调度器和优化器
    # 制定定步长学习率衰减策略
    lr_scheduler = paddle.optimizer.lr.LambdaDecay(
        LR,
        lr_lambda = lambda x : 1-x/(15126)
    )
    # 构造Adam优化器
    optimizer = paddle.optimizer.Adam(
        learning_rate=lr_scheduler,
        parameters=model.net.parameters()
    )
    # 调用PaddleRS API实现一键训练
    model.train(
        num_epochs=NUM_EPOCHS,
        train_dataset=train_dataset,
        train_batch_size=TRAIN_BATCH_SIZE,
        eval_dataset=eval_dataset,
        optimizer=optimizer,
        save_interval_epochs=SAVE_INTERVAL_EPOCHS,
        # 每多少次迭代记录一次日志
        log_interval_steps=10,  
        save_dir=EXP_DIR,
        # 是否使用early stopping策略，当精度不再改善时提前终止训练
        early_stop=False,
        # 是否启用VisualDL日志功能
        use_vdl=True,
        # 指定从某个检查点继续训练
        resume_checkpoint=None
    )
