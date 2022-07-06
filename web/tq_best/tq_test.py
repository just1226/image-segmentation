from paddlers.deploy.predictor import Predictor
import numpy as np
import matplotlib
from PIL import Image
from matplotlib import pyplot as plt
import cv2
# import export
matplotlib.use('TkAgg')

""" 
第一步：构建Predictor。该类接受的构造参数如下：
    model_dir: 模型路径（必须是导出的部署或量化模型）。
    use_gpu: 是否使用GPU，默认为False。
    gpu_id: 使用GPU的ID，默认为0。
    cpu_thread_num：使用cpu进行预测时的线程数，默认为1。
    use_mkl: 是否使用mkldnn计算库，CPU情况下使用，默认为False。
    mkl_thread_num: mkldnn计算线程数，默认为4。
    use_trt: 是否使用TensorRT，默认为False。
    use_glog: 是否启用glog日志, 默认为False。
    memory_optimize: 是否启动内存优化，默认为True。
    max_trt_batch_size: 在使用TensorRT时配置的最大batch size，默认为1。
    trt_precision_mode：在使用TensorRT时采用的精度，可选值['float32', 'float16']。默认为'float32'。
"""
    # 下面的语句构建的Predictor对象依赖static_models/1024x1024中存储的部署格式模型，并使用GPU进行推理。
predictor = Predictor('./tq_best/export', use_gpu=False)

""" 
第二步：调用Predictor的predict()方法执行推理。该方法接受的输入参数如下：
    img_file(List[str or tuple or np.ndarray], str, tuple, or np.ndarray):
        对于场景分类、图像复原、目标检测和语义分割任务来说，该参数可为单一图像路径，或是解码后的、排列格式为（H, W, C）
        且具有float32类型的BGR图像（表示为numpy的ndarray形式），或者是一组图像路径或np.ndarray对象构成的列表；对于变化检测
        任务来说，该参数可以为图像路径二元组（分别表示前后两个时相影像路径），或是两幅图像组成的二元组，或者是上述两种二元组
        之一构成的列表。
    topk(int): 场景分类模型预测时使用，表示预测前topk的结果。默认值为1。
    transforms (paddlers.transforms): 数据预处理操作。默认值为None, 即使用`model.yml`中保存的数据预处理操作。
    warmup_iters (int): 预热轮数，用于评估模型推理以及前后处理速度。若大于1，会预先重复预测warmup_iters，而后才开始正式的预测及其速度评估。默认为0。
    repeats (int): 重复次数，用于评估模型推理以及前后处理速度。若大于1，会预测repeats次取时间平均值。默认值为1。
"""
# 下面的语句传入两幅输入影像的路径
res = predictor.predict("./saved_files/tq/img.png")
print(res)

""" 
第三步：解析predict()方法返回的结果。
    对于语义分割和变化检测任务而言，predict()方法返回的结果为一个字典或字典构成的列表。字典中的`label_map`键对应的值为类别标签图，对于二值变化检测
    任务而言只有0（不变类）或者1（变化类）两种取值；`score_map`键对应的值为类别概率图，对于二值变化检测任务来说一般包含两个通道，第0个通道表示不发生
    变化的概率，第1个通道表示发生变化的概率。如果返回的结果是由字典构成的列表，则列表中的第n项与输入的img_file中的第n项对应。
"""
# 下面的语句从res中解析二值变化图（binary change map）n
cm = res['label_map']

# 从左到右依次显示：第一时相影像、第二时相影像、整图推理结果以及真值标签
plt.figure(constrained_layout=True)
plt.subplot(121)
plt.imshow(Image.open("./saved_files/tq/img.png"))
plt.gca().set_axis_off()
plt.title("Image1")
plt.subplot(122)
# plt.imshow(Image.open("img/B.png"))
# plt.gca().set_axis_off()
# plt.title("Image2")
# plt.subplot(133)
plt.imshow((cm*255).astype('uint8'))
plt.gca().set_axis_off()
plt.title("Pred")
cv2.imwrite("./saved_imgs/tq/test.png",(cm*255).astype('uint8'))
plt.savefig('./saved_imgs/tq/tq_res.png')
#plt.subplot(144)
#plt.imshow((np.asarray(Image.open("data/data134796/dataset/train/label/train_13.png"))*255).astype('uint8'))
#plt.gca().set_axis_off()
#plt.title("GT")

# plt.show()