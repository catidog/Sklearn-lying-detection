from fall_dataset.action_data import ActionData
from configs.fall_config import Generate_Para

if __name__ == '__main__':
    cfg = Generate_Para()
    data = ActionData(cfg)
    data.generate_action_data()