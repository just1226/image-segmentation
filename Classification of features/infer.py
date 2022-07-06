import random
import os.path as osp

import cv2
import numpy as np
import paddle
import paddlers as pdrs
from matplotlib import pyplot as plt
from PIL import Image
from train import model, eval_transforms

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
# 构建测试集
test_dataset = pdrs.datasets.SegDataset(
    data_dir=DATA_DIR,
    file_list=TEST_FILE_LIST_PATH,
    label_list=LABEL_LIST_PATH,
    transforms=eval_transforms,
    num_workers=0,
    shuffle=False
)


# 为模型加载历史最佳权重
state_dict = paddle.load(osp.join(EXP_DIR, 'best_model/model.pdparams'))
model.net.set_state_dict(state_dict)

# 执行测试
test_result = model.evaluate(test_dataset)
print(
    "测试集上指标：mIoU为{:.2f}，OAcc为{:.2f}，Kappa系数为{:.2f}".format(
        test_result['miou'],
        test_result['oacc'],
        test_result['kappa'],
    )
)

print("各类IoU分别为："+', '.join('{:.2f}'.format(iou) for iou in test_result['category_iou']))
print("各类Acc分别为："+', '.join('{:.2f}'.format(acc) for acc in test_result['category_acc']))
print("各类F1分别为："+', '.join('{:.2f}'.format(f1) for f1 in test_result['category_F1-score']))

# 预测结果可视化
# 重复运行本单元可以查看不同结果

def show_images_in_row(ims, fig, title='', lut=None):
    n = len(ims)
    fig.suptitle(title)
    axs = fig.subplots(nrows=1, ncols=n)
    for idx, (im, ax) in enumerate(zip(ims, axs)):
        # 去掉刻度线和边框
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.get_xaxis().set_ticks([])
        ax.get_yaxis().set_ticks([])

        if isinstance(im, str):
            im = cv2.imread(im, cv2.IMREAD_COLOR)
        if lut is not None:
            if im.ndim == 3:
                im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
            im = lut[im]
        else:
            im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        ax.imshow(im)


def get_lut():
    lut = np.zeros((256,3), dtype=np.uint8)
    lut[0] = [255, 0, 0]
    lut[1] = [30, 255, 142] #林地
    lut[2] = [60, 0, 255] #道路
    lut[3] = [255, 222, 0]
    lut[4] = [0, 0, 0] #其他
    return lut


# 需要展示的样本个数
num_imgs_to_show = 4
# 随机抽取样本
chosen_indices = random.choices(range(len(test_dataset)), k=num_imgs_to_show)

# 参考 https://stackoverflow.com/a/68209152
fig = plt.figure(constrained_layout=True)
fig.suptitle("Test Results")

subfigs = fig.subfigures(nrows=3, ncols=1)

# 读取输入影像并显示
im_paths = [test_dataset.file_list[idx]['image'] for idx in chosen_indices]
show_images_in_row(im_paths, subfigs[0], title='Image')

# 获取模型预测输出
with paddle.no_grad():
    model.net.eval()
    preds = []
    for idx in chosen_indices:
        input, _ = test_dataset[idx]
        input = paddle.to_tensor(input).unsqueeze(0)
        logits, *_ = model.net(input)
        pred = paddle.argmax(logits[0], axis=0)
        pred = pred.numpy().astype(np.uint8)
        preds.append(pred)
show_images_in_row(preds, subfigs[1], title='Pred', lut=get_lut())

# 读取真值标签并显示
im_paths = [test_dataset.file_list[idx]['mask'] for idx in chosen_indices]
show_images_in_row(im_paths, subfigs[2], title='GT', lut=get_lut())

# 渲染结果
fig.canvas.draw()
Image.frombytes('RGB', fig.canvas.get_width_height(), fig.canvas.tostring_rgb())
