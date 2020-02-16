
class Parameters():
    def __init__(self):
        self.input_path = '/media/deepnorth/Backup_Deep/jinliang/action/fall-dataset/labels'

        # option: Nearest Neighbors, Linear SVM, RBF SVM,
        # Gaussian Process, Decision Tree, Random Forest, Neural Net,
        # AdaBoost, Naive Bayes, QDA
        self.model = 'Neural Net'
        self.classes = ['nofall', 'fall']
        self.output_path = '/home/jinliang/work/action/fall-recognition/' \
                           'trained_models/MLP_classifier_n5_w15_hf.pickle'

        self.pca = False
        self.pca_feature_num = 100
        self.norm = 5

class Generate_Para():
    def __init__(self):
        self.class_list = ['nofall', 'fall']
        self.class_dic = {'nofall': 0,
                          'fall': 1}

        self.window_size = 15
        self.keypoint_num = 25 # 没有用

        self.norm = 5
        self.skip_frame = 2
        self.split = 0.8
        self.delete_headfoot = False

        self.json_folder = '/media/deepnorth/Backup_Deep/' \
                           'jinliang/action/fall-dataset/skeletons'
        self.out_folder = '/media/deepnorth/Backup_Deep/' \
                          'jinliang/action/fall-dataset/labels'
