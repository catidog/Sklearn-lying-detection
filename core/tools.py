import numpy as np

def read_data(input_path):
    '''
        读取txt文件，每行是array的一个元素
    :param input_path:
    :return:
    '''
    data = []
    with open(input_path, 'r') as f:
        lines = f.readlines()

    # str => np.array 删掉所有[],将数字用，隔开
    for i in range(len(lines)):

        tmp_data = lines[i].replace('[','')
        tmp_data = tmp_data.replace(']','')
        tmp_data = tmp_data.split(',')
        tmp_data = list(map(float, tmp_data))
        data.append(tmp_data)

    return np.array(data)