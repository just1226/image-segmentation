import codecs
import os
from collections import Counter
import yaml
from numpy import array,concatenate
import paddleseg.transforms as T
from paddle.inference import create_predictor, PrecisionType
from paddle.inference import Config as PredictConfig
from paddleseg.cvlibs import manager
from paddleseg.utils import get_sys_env
from paddleseg.utils.visualize import get_pseudo_color_map
from cv2 import imread
from base64 import b64decode,b64encode
from io import BytesIO
import json
# from Qscore_algorithm import cal_Q_score
import time

env_info = get_sys_env()
use_gpu = True if env_info['Paddle compiled with cuda'] and env_info['GPUs used'] else False

batch_size = 1

class DeployConfig:
    '''
    加载模型信息文件，里面可以设置transforms
    这里需要提醒一下，在实际生产环境下，用户的CPU和显卡配置未知，容易导致预测过程中程序进程因为占用系统资源过多而被杀掉
    因此推荐手动在配置文件中最好设置一下transforms，对输入图像进行resize或者等比缩放，减少运算量
    在github上项目的配置文件就加上了最长边等比缩放，加上了最大为1920像素,最小为500像素的限制。

    transforms:
        - type: ResizeRangeScaling
        min_value: 500
        max_value: 1920

    '''
    def __init__(self, path):
        with codecs.open(path, 'r', 'utf-8') as file:
            self.dic = yaml.load(file, Loader=yaml.FullLoader)

        self._transforms = self._load_transforms(
            self.dic['Deploy']['transforms'])
        self._dir = os.path.dirname(path)

    @property
    def transforms(self):
        return self._transforms

    @property
    def model(self):
        return os.path.join(self._dir, self.dic['Deploy']['model'])

    @property
    def params(self):
        return os.path.join(self._dir, self.dic['Deploy']['params'])

    def _load_transforms(self, t_list):
        com = manager.TRANSFORMS
        transforms = []
        for t in t_list:
            ctype = t.pop('type')
            transforms.append(com[ctype](**t))

        return T.Compose(transforms)

class Predictor:
    def __init__(self):
        # self.cfg = DeployConfig('./export/model.yml') #这里需要修改成模型配置文件
        # yaml_path = r'./export/model.yml'
        
            # self.cfg={}
            # yaml.dump(self.cfg,f)
            # print(self.cfg)
            # print(self.cfg['transforms'])
    # pred_cfg = PredictConfig(self.cfg.model, self.cfg.params)
        pred_cfg = PredictConfig('./export/model.pdmodel','./export/model.pdiparams')
        pred_cfg.disable_gpu()
        pred_cfg.disable_glog_info()
        # if use_gpu:
        #     pred_cfg.enable_use_gpu(100, 0)
        #     ptype = PrecisionType.Float32

        self.predictor = create_predictor(pred_cfg) #传入参数到create_predictor函数

    def preprocess(self, img): #数据预处理
        return self.cfg.transforms(img)[0]

    def run(self, imgs):
        if not isinstance(imgs, (list, tuple)):
            imgs = [imgs]

        num = len(imgs)
        input_names = self.predictor.get_input_names()
        input_handle = self.predictor.get_input_handle(input_names[0])
        results = []

        for i in range(0, num, batch_size):
            data = array([
                self.preprocess(img) for img in imgs[i:i + batch_size]
            ])
            input_handle.reshape(data.shape)
            input_handle.copy_from_cpu(data)
            self.predictor.run()

            output_names = self.predictor.get_output_names()
            output_handle = self.predictor.get_output_handle(output_names[0])
            results.append(output_handle.copy_to_cpu())

        result = self.postprocess(results, imgs) #将预测结果传到postprocess函数处理
        return result

    def postprocess(self, results, imgs):

        results = concatenate(results, axis=0)
        for i in range(results.shape[0]):
            result = results[i] #接收预测结果
            #把预测结果的矩阵折叠成一维数组并统计每一类数量
            pix_count = dict(Counter(result.flatten()))
            pix_sum = sum(pix_count.values()) #统计像素总数量

            #循环计算每一类要素占总像素的百分比
            # cityscape_class = ['road','sidewalk','building',
            #                    'wall','fence','pole','traffic light',
            #                    'traffic sign','vegetation','terrain',
            #                    'sky','person','rider','car','truck',
            #                    'bus','train','motorcycle','bicycle']
            CLASSES = (
                'background',
                'road',
            )
            element_percentage = {'sky':0, 'terrain':0, 'vegetation':0,
                                  'pole':0, 'fence':0, 'wall':0, 'building':0,
                                  'sidewalk':0, 'road':0, 'others':0}
            for c in range(0,len(CLASSES)):
                try:
                    if element_percentage.__contains__(CLASSES[c]):
                        element_percentage[CLASSES[c]]=pix_count[c]/pix_sum
                    else:
                        element_percentage['others'] += pix_count[c]/pix_sum
                except:pass
            #计算绿视率
            GLR = round(element_percentage['vegetation']+element_percentage['terrain'],2)
            #计算分数
            # Q_score = json.dumps(cal_Q_score(element_percentage))
            #字典转json
            element_percentage = json.dumps(element_percentage)
            #渲染彩色效果图
            result = get_pseudo_color_map(result)
            result = result.convert('RGB')
            #临时保存图片
            buffered = BytesIO()
            result.save(buffered, format="JPEG")
            result = b64encode(buffered.getvalue()).decode('utf-8')

            return element_percentage,result,GLR #返回百分比/彩色图/各项得分/绿视率

import json

def seg_photo(image_path):
    predictor = Predictor()
    result = predictor.run([image_path])  #传入图片
    return result

def seg_api(image_base64):
    image_post = b64decode(image_base64)  #接收图片base64
    #将base64转换成临时图片文件
    if not os.path.exists('./tmp'): os.mkdir('./tmp')
    tmp_file = './tmp/{}.jpg'.format(time.time())
    with open(tmp_file,'wb') as f:
        f.write(image_post)
    seg_result = seg_photo(tmp_file)  #对图片文件进行预测
    os.unlink(tmp_file)
    return seg_result  #返回JSON格式化后的预测结果
import json

test_img = open("../saved_files/img.png","rb") #打开图片
base64_data = b64encode(test_img.read()).decode('utf-8') #转换成base64
result = seg_api(base64_data) #向api函数传入数据
print(result) #返回JSON结果