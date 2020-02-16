import os
import time
import joblib
from core.classifier import ClassifierOfflineTrain
from core.log import setup_logger
from core.tools import read_data
from configs.fall_config import Parameters

if __name__ == '__main__':
    logger = setup_logger(name='train action model')
    para = Parameters()

    # -- Load preprocessed data
    logger.info("Reading txt files of classes and skeleton data...")
    train_data_txt = os.path.join(para.input_path, 'train_norm{}_data.txt'.format(para.norm))
    train_label_txt = os.path.join(para.input_path, 'train_norm{}_label.txt'.format(para.norm))

    X = read_data(train_data_txt)  # data
    Y = read_data(train_label_txt).T[0] # labels y必须是1维
    logger.info("Size of training data X: {}".format(X.shape))

    # -- Train the model
    logger.info("Start training model ...")
    start = time.time()
    model = ClassifierOfflineTrain(para)
    model.train(X, Y)
    end = time.time() - start
    logger.info('Train cost {:.2f}s'.format(end))

    # -- Evaluate model
    logger.info("Start evaluating model...")
    val_data_txt = os.path.join(para.input_path, 'val_norm{}_data.txt'.format(para.norm))
    val_label_txt = os.path.join(para.input_path, 'val_norm{}_label.txt'.format(para.norm))
    val_X = read_data(val_data_txt)
    val_Y = read_data(val_label_txt).T[0]

    model.evaluate_model(model, X, Y, val_X, val_Y)
    logger.info('Evaluation end.')

    # -- Save model
    logger.info("Save model to {}".format(para.output_path))
    joblib.dump(model.clf, para.output_path)
    # with open(para.output_path, 'wb') as f:
    #     pickle.dump(model, f)