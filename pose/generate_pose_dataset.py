'''
    解析box,将图像里每个box作为图像存放, 存储图像文件名格式：
        index_类别名.png

    把图像分成train/test两个部分，每个对应的绝对路径存到train.txt, 和test.txt里
'''
from core.log import setup_logger
from pose.xml_label import Xml_Reader
from configs.pose_config import Pose_Parameters

logger = setup_logger(name='pose box dataset')
cfg = Pose_Parameters

# image => box
logger.info('Start crop image to box image.....')

input_data = Xml_Reader(cfg=cfg)
input_data.xml_to_box()

logger.info('box image count: {}'.format(input_data.box_count))