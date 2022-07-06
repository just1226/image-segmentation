from models import model, eval_transforms
import paddle
import paddlers as pdrs
import os.path as osp
import cv2
import random
from matplotlib import pyplot as plt
from PIL import Image
from paddlers.tasks.utils.visualize import visualize_detection
import numpy as np

# 数据集存放目录
DATA_DIR = '/home/aistudio/data/data52980/rsod/'
# 训练集`file_list`文件路径
TRAIN_FILE_LIST_PATH = '/home/aistudio/data/data52980/rsod/train.txt'
# 验证集`file_list`文件路径
VAL_FILE_LIST_PATH = '/home/aistudio/data/data52980/rsod/val.txt'
# 测试集`file_list`文件路径
TEST_FILE_LIST_PATH = '/home/aistudio/data/data52980/rsod/test.txt'
# 数据集类别信息文件路径
LABEL_LIST_PATH = '/home/aistudio/data/data52980/rsod/labels.txt'
# 实验目录，保存输出的模型权重和结果
EXP_DIR =  '/home/aistudio/exp/'
# 目标类别
CLASS = 'playground'
# 模型验证阶段输入影像尺寸
INPUT_SIZE = 608
# 构建测试集
test_dataset = pdrs.datasets.VOCDetection(
    data_dir=DATA_DIR,
    file_list=TEST_FILE_LIST_PATH,
    label_list=LABEL_LIST_PATH,
    transforms=eval_transforms,
    shuffle=False
)


# 为模型加载历史最佳权重
state_dict = paddle.load(osp.join(EXP_DIR, 'epoch_100/model.pdparams'))
model.net.set_state_dict(state_dict)

# 执行测试
test_result = model.evaluate(test_dataset)
print(
    "测试集上指标：bbox mAP为{:.2f}".format(
        test_result['bbox_map'],
    )
)

# 预测结果可视化
# 重复运行本单元可以查看不同结果

def read_rgb(path):
    im = cv2.imread(path)
    im = im[...,::-1]
    return im


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

        ax.imshow(im)


# 需要展示的样本个数
num_imgs_to_show = 4
# 随机抽取样本
chosen_indices = random.choices(range(len(test_dataset)), k=num_imgs_to_show)

# 参考 https://stackoverflow.com/a/68209152
fig = plt.figure(constrained_layout=True)
fig.suptitle("Test Results")

subfigs = fig.subfigures(nrows=2, ncols=1)

# 读取输入影像并显示
ims = [read_rgb(test_dataset.file_list[idx]['image']) for idx in chosen_indices]
show_images_in_row(ims, subfigs[0], title='Image')

# 绘制目标框
with paddle.no_grad():
    vis_res = []
    model.labels = test_dataset.labels
    for idx, im in zip(chosen_indices, ims):
        sample = test_dataset[idx]
        gt = [
            {
                'category_id': cid[0],
                'category': CLASS,
                'bbox': [bbox[0], bbox[1], bbox[2]-bbox[0], bbox[3]-bbox[1]],
                'score': 1.0
            }
            for cid, bbox in zip(sample['gt_class'], sample['gt_bbox'])
        ]

        im = cv2.resize(im[...,::-1], (INPUT_SIZE, INPUT_SIZE), interpolation=cv2.INTER_CUBIC)
        pred = model.predict(im, eval_transforms)

        vis = im
        # 用绿色画出预测目标框
        if len(pred) > 0:
            vis = visualize_detection(
                np.array(vis), pred,
                color=np.asarray([[0,255,0]], dtype=np.uint8),
                threshold=0.2, save_dir=None
            )
        # 用蓝色画出真实目标框
        if len(gt) > 0:
            vis = visualize_detection(
                np.array(vis), gt,
                color=np.asarray([[0,0,255]], dtype=np.uint8),
                save_dir=None
            )
        vis_res.append(vis)
show_images_in_row(vis_res, subfigs[1], title='Detection')

# 渲染结果
fig.canvas.draw()
Image.frombytes('RGB', fig.canvas.get_width_height(), fig.canvas.tostring_rgb())
