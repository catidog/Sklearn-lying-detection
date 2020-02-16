# coding:utf-8

"""
    对骨架坐标进行归一化处理,无法得到图像的宽和高
"""

import numpy as np

# HIP_JOINT = (16,17)
HIP_JOINT = (14,15)

def list_norm(input):
    """
        对一个list进行白化，return仍然为list
        需要考虑没有检测到的点坐标为（0，0）的情况
    :param input:
    :return:
    """
    res = list(filter(lambda x:int(x)!=0, input)) # 除去0的list

    input_mean = np.mean(np.array(res))
    input_var = np.std(np.array(res))

    input_norm = [abs(i-input_mean)/input_var if int(i) != 0 else i for i in input ]
    return input_norm

def coord_transfomation2D(point, matrix):
    '''
       function: 求2D A坐标系变换到B坐标系的变换矩阵

       :param point: B坐标原点 在A坐标系的位置
       :param matrix: B坐标系的基向量 在A坐标系里表示 B = [u_x, u_y, u_z]
       :return: 由A坐标系 变换到B坐标系的变换矩阵 4*4
    '''

    Tx = point[0]
    Ty = point[1]

    T = np.array([[matrix[0][0], matrix[1][0], Tx],
                  [matrix[0][1], matrix[1][1], Ty],
                  [0, 0, 1]])

    return T

def point_coordTrans(point, new_Axis_origin, Axis):
    '''
        function: 由point在原始坐标系的坐标,计算point在new_Axis下的坐标, 需要扩展一个维度

        :param point: point在原始坐标系的坐标
        :param new_Axis_origin: new_Axis的原点在原始坐标系的位置
        :param Axis: new_Axis的基向量在原始坐标系下的表示
        :return: point在new_Axis下的坐标new_point
    '''

    # step 0: point坐标扩展一个维度
    point.append(1)

    # step 1: 求坐标系变换矩阵
    Trans_mat = np.linalg.inv(coord_transfomation2D(new_Axis_origin, Axis))

    # step 2: 坐标变换
    new_point = np.dot(Trans_mat, point)
    point.pop()
    return list(new_point[:-1])

def pair_data(ske):
    """
        将一维list[x0,y0,x1,y1...]变成[[x0,y0],[x1,y1],...]
    :param ske:
    :return:
    """
    if len(ske) % 2 != 0:
        print('data length is not 2n')
        return False
    out = []
    q = []
    for i in range(len(ske)):
        q.append(ske[i])
        if len(q) == 2:
            out.append(q)
            q = []
    return out

def unpair_data(ske):
    """
        将[[x0,y0],[x1,y1],...]变成一维list[x0,y0,x1,y1...]
    :param ske:
    :return:
    """
    out = []
    for i in range(len(ske)):
        out += ske[i]
    return out

def data_normalization(skeleton, mode=0):
    """
        mode为归一化方式,默认为0不做归一化
        mode1：减去均值，除以方差

        mode2：除以最大值
        mode3：减去最小值，除以最大减去最小

        mode4：所有骨头长度除以躯干

        mode5：相对于hip的坐标

    :param skeleton: list 50维度
    :param mode:
    :return:
    """

    if mode != 0:
        x = skeleton[0::2]
        y = skeleton[1::2]

        if mode == 1:
            x_norm = list_norm(x)
            y_norm = list_norm(y)

        elif mode == 2:
            max_x = max(x)
            max_y = max(y)
            x_norm = map(lambda x:x/max_x, x)
            y_norm = map(lambda x:x/max_y, y)

        elif mode == 3:
            x_norm = map(lambda x:(x-min(x))/(max(x)-min(x)), x)
            y_norm = map(lambda x:(x-min(y))/(max(y)-min(y)), y)

        elif mode == 5:
            """
                2D全局坐标转换为局部坐标
            """
            hip_point = [skeleton[HIP_JOINT[0]]+1, skeleton[HIP_JOINT[1]]+1] # hip 附近的点作为坐标原点
            hip_axis = [[-1, 0],
                        [0, -1]]

            ske_pair = pair_data(skeleton)

            norm_data = []
            for i in range(len(ske_pair)):
                a = [int(ske_pair[i][0]), int(ske_pair[i][1])]
                if a != [0,0]:
                    temp = point_coordTrans(ske_pair[i], hip_point, hip_axis)
                else:
                    temp = ske_pair[i]
                norm_data.append(temp)

            norm_data = unpair_data(norm_data)
            x_norm = norm_data[::2]
            y_norm = norm_data[1::2]

        tmp = (x_norm, y_norm)
        out = [tmp[i % 2].pop(0) if tmp[i % 2] else tmp[1 - i % 2].pop(0) for i in range(len(skeleton))]
    else:
        out = skeleton

    return out