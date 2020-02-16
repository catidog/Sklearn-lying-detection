'''
    解析xml文件里的每个box的位置以及类别 xml的框的坐标是没有归一化的坐标
    每张图像对应一个xml文件，两个的文件名是相同的
'''
import os
import cv2
from tqdm import tqdm
import xml.etree.ElementTree as ET

class Xml_Reader():
    def __init__(self, cfg):
        self.cfg = cfg
        self.folder_path = self.cfg.image_folder
        self.class_list = self.cfg.class_list
        self.xml_layer = self.cfg.xml_layer

        self.class_dic = {}
        for i in range(len(self.class_list)):
            self.class_dic[i] = self.class_list[i]

        self.xml_path_list = self.read_folder()
        self.box_count = 0
        self.output_path = self.cfg.output_path
        self.img_format = self.cfg.image_format

    def read_folder(self):
        """
            fuction: put all the xml files into a list
        :return:
        """
        file_pathlist = os.listdir(os.path.join(self.folder_path, 'gtXml'))
        filepathlist = []
        for filepath in file_pathlist:
            filepath1 = os.path.join(self.folder_path, 'gtXml', filepath)
            filepathlist.append(filepath1)
        return filepathlist

    def get_staff_box(self, tree):
        root = tree.getroot()
        nodes = root.getchildren()
        class_and_bboxs = []

        for node in nodes:
            child_nodes = node.getchildren()
            class_bbox = {'name': 'label_name',
                          'xmin': 0,
                          'ymin': 0,
                          'xmax': 0,
                          'ymax': 0}

            for eles in child_nodes:
                if (eles.tag == 'name'):
                    class_bbox['name'] = eles.text

                if (eles.tag == 'bndbox'):
                    for ele in eles:
                        if (ele.tag == 'xmin'):
                            class_bbox['xmin'] = float(ele.text)
                        if (ele.tag == 'ymin'):
                            class_bbox['ymin'] = float(ele.text)
                        if (ele.tag == 'xmax'):
                            class_bbox['xmax'] = float(ele.text)
                        if (ele.tag == 'ymax'):
                            class_bbox['ymax'] = float(ele.text)
            class_and_bboxs.append(class_bbox)
        return class_and_bboxs
    
    def check_box(self, image_shape, box): # box [x1, y1, x2, y2]
        '''
            check box
        '''
        image_h = image_shape[0] # y 720
        image_w = image_shape[1] # x 1280
        
        out = []
        for i in range(len(box)):
            if box[i] < 0:
                box[i] = 0
            if i%2 == 0 and box[i] > image_w: # x1, x2
                box[i] = image_w
            if i%2 == 1 and box[i] > image_h:
                box[i] = image_h
            out.append(box[i])
        return out

    def xml_to_box(self):

        for filepath in tqdm(self.xml_path_list):
            write_xml_filename = os.path.split(filepath)[-1]
            image_filepath = os.path.join(self.folder_path,
                                          os.path.splitext(write_xml_filename)[0] + '.jpg') # xml对应的图像的路径

            tree = ET.parse(filepath)
            img = cv2.imread(image_filepath)
            class_and_bboxs = self.get_staff_box(tree)

            # print(class_and_bboxs)
            for class_and_bbox in class_and_bboxs:
                x1 = int(class_and_bbox['xmin'])
                y1 = int(class_and_bbox['ymin'])
                x2 = int(class_and_bbox['xmax'])
                y2 = int(class_and_bbox['ymax'])
                box = self.check_box(img.shape, 
                                     [x1, y1, x2, y2])

                class_name = class_and_bbox['name']

                crop_img = img[box[1]:box[3], box[0]:box[2], :] # h w c

                tmp_output_name = '{0}_{1}.{2}'.format(self.box_count,
                                                        class_name,
                                                        self.img_format)

                if not os.path.exists(self.output_path):
                    os.mkdir(self.output_path)

                tmp_output_path = os.path.join(self.output_path, tmp_output_name)
                cv2.imwrite(tmp_output_path, crop_img)
                self.box_count += 1


