'''
    将视频里每帧的json组合成几个clip，每行是一个clip的数据:
        clip1: [ske1, ske2, ...]
        clip2: [ske1, ske2, ...] ...

    json文件命名格式：文件名_rgb_000000000129_keypoints.json 12位

    label_txt每行是标签
        clip1: label_index
        clip2: label_index ...
'''
import os
import json
import random

from core.log import setup_logger
from tqdm import tqdm
from fall_dataset.normalize import data_normalization

class ActionData():
    def __init__(self, cfg):
        self.logger = setup_logger(name='action data')
        self.cfg = cfg
        self.dump_count = 0
        self.class_list = self.cfg.class_list
        self.class_dic = self.cfg.class_dic
        self.window_size = self.cfg.window_size

        self.skeleton_folder = self.cfg.json_folder
        self.norm = self.cfg.norm
        self.skip_frame = self.cfg.skip_frame
        self.split = self.cfg.split

        self.delete_headfoot = self.cfg.delete_headfoot

        # output
        self.skeletons = []
        self.label = []
        self.output = self.cfg.out_folder
        if not os.path.exists(self.output):
            os.mkdir(self.output)

        self.train_count = 0
        self.val_count = 0

        self.train_label_txt = os.path.join(self.output, 'train_norm{}_label.txt'.format(self.norm))
        self.train_data_txt = os.path.join(self.output, 'train_norm{}_data.txt'.format(self.norm))
        self.val_label_txt = os.path.join(self.output, 'val_norm{}_label.txt'.format(self.norm))
        self.val_data_txt = os.path.join(self.output, 'val_norm{}_data.txt'.format(self.norm))

    def load_person(self, input_path):
        json_data = json.load(open(input_path, 'r'))
        people = json_data['people']
        return people

    def dump_no_skeleton(self, video):
        """
            去除没有检测到骨架的帧，输出为有骨架的按顺序的帧号的list 如果总帧数小于num_frame 直接输出False
        :param video_path:
        :return: 按顺序的帧序号
        """
        j_list = os.listdir(video)
        length = len(j_list) # 帧数

        # 找到所有没有检测的帧
        no_skeleton_index = []  # 记录没有检测的帧号

        for j_file in j_list:
            frame_i = int(j_file.split('_')[-2])  # 文件名 v_TouchHigh_100_000000000000_keypoints.json

            j_path = os.path.join(video, j_file)
            v_people = self.load_person(j_path)
            if v_people == []:
                no_skeleton_index.append(frame_i)

        # 从剔除的之外，按照帧序列随机选择
        big_list = list(range(length))
        small_list = no_skeleton_index
        choice_list = []

        for i in big_list:
            if i not in small_list:
                choice_list.append(i)
            else:
                pass
        return sorted(choice_list)

    def clip_skeleton(self, index, choice, vid_path):
        '''
            index是起始序号，window_size的clip里的所有skeleton
        :param index_list:
        :return:
        '''
        clip_skeleton = []
        for i in range(index, index + self.window_size):
            frame_index = str(choice[i]).zfill(12)
            video_name = os.path.basename(vid_path)
            tmp_json_name = '{0}_{1}_keypoints.json'.format(video_name,
                                                            frame_index) # frame_index => json name
            json_path = os.path.join(vid_path, tmp_json_name)
            v_people = self.load_person(json_path)
            person_pose_keypoints = v_people[0]['pose_keypoints_2d']
            final_pose_keypoints = [item for sublist in
                                    zip(person_pose_keypoints[0::3], person_pose_keypoints[1::3]) for item
                                    in
                                    sublist]

            if self.delete_headfoot == True:
                # 去除头脚的关节点  0，11，14-24 X,Y—— 0,1 22,23 28-49
                final_pose_keypoints = final_pose_keypoints[:28]
                final_pose_keypoints.pop(0) # list删去一个元素后，位置发生变化
                final_pose_keypoints.pop(0)
                final_pose_keypoints.pop(20)
                final_pose_keypoints.pop(20)

            # 归一化方法 删除投肩后归一化需要调整才行
            final_pose_keypoints = data_normalization(final_pose_keypoints, mode=self.norm)
            clip_skeleton.append(final_pose_keypoints)
        return clip_skeleton

    def write_output(self):
        '''
            把skeleton和label写入txt, 将数据分为train, val
        :return:
        '''
        # write data label
        with open(self.train_data_txt, 'w+') as train_data:
            with open(self.train_label_txt, 'w+') as train_label:
                with open(self.val_data_txt, 'w+') as val_data:
                    with open(self.val_label_txt, 'w+') as val_label:

                        for i in range(len(self.skeletons)):
                            p = random.random()

                            if p <= self.split:
                                self.train_count += 1
                                train_data.write(str(self.skeletons[i]) + '\n')
                                train_label.write(str(self.label[i]) + '\n')
                            else:
                                self.val_count += 1
                                val_data.write(str(self.skeletons[i]) + '\n')
                                val_label.write(str(self.label[i]) + '\n')

    def generate_action_data(self):
        for cls in self.class_list:
            tmp_cls_path = os.path.join(self.skeleton_folder, cls)
            tmp_video_list = os.listdir(tmp_cls_path)

            for vid in tqdm(tmp_video_list, desc='start writing ' + cls + ' skeleton'):
                tmp_vid_path = os.path.join(tmp_cls_path, vid)
                choice_index = self.dump_no_skeleton(tmp_vid_path)

                # 视频里能用的帧小于window_size就不用这个视频了
                if len(choice_index) <= self.window_size:
                    self.dump_count += 1
                    continue
                else:
                    for s_index in range(0, len(choice_index)-self.window_size,
                                         self.skip_frame):
                        tmp_clip_skeleton = self.clip_skeleton(s_index,
                                                               choice_index,
                                                               tmp_vid_path)

                        self.skeletons.append(tmp_clip_skeleton)
                        self.label.append(self.class_dic[cls])
        # write txt
        self.write_output()

        self.logger.info('Dumped json file number: {}'.format(self.dump_count))
        self.logger.info('Skeleton data clip number: {}'.format(len(self.skeletons)))
        self.logger.info('Label data clip number: {}'.format(len(self.label)))
        self.logger.info('Train clip number: {}'.format(self.train_count))
        self.logger.info('Val clip number: {}'.format(self.val_count))
