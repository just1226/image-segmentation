# 划分训练集/验证集，并生成文件名列表

import random
import os.path as osp
from glob import glob


# 随机数生成器种子
RNG_SEED = 114514
# 调节此参数控制训练集数据的占比
TRAIN_RATIO = 0.90
# 数据集路径
DATA_DIR = '/BIT-CD-data'


def write_rel_paths(phase, names, out_dir, prefix=''):
    """将文件相对路径存储在txt格式文件中"""
    with open(osp.join(out_dir, phase+'.txt'), 'w') as f:
        for name in names:
            f.write(
                ' '.join([
                    osp.join(prefix, 'A', name),
                    osp.join(prefix, 'B', name),
                    osp.join(prefix, 'label', name)
                ])
            )
            f.write('\n')


random.seed(RNG_SEED)

# 随机划分训练集/验证集
names = list(map(osp.basename, glob(osp.join(DATA_DIR, 'train', 'label', '*.png'))))
# 对文件名进行排序，以确保多次运行结果一致
names.sort()
random.shuffle(names)
len_train = int(len(names)*TRAIN_RATIO) # 向下取整
write_rel_paths('train', names[:len_train], DATA_DIR, prefix='train')
write_rel_paths('val', names[len_train:], DATA_DIR, prefix='train')

# 处理测试集
test_names = map(osp.basename, glob(osp.join(DATA_DIR, 'test', 'A', '*.png')))
test_names = sorted(test_names)
write_rel_paths(
    'test', 
    test_names, 
    DATA_DIR,
    prefix='test'
)

print("数据集划分已完成。")
