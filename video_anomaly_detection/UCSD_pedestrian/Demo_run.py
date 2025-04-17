import yaml
from sklearn.metrics import roc_auc_score
from IPython.display import HTML

from Dataset.UCSD_dataset import load_test_data,load_train_data
from DMAD.run import DMAD

LOAD_PATH = '/ssd003/projects/aieng/public/anomaly_detection_models/UCSD/'
TRAIN_DATA_PATH = '/ssd003/projects/aieng/public/anomaly_detection_datasets/UCSD_Anomaly_Dataset/UCSDped2/Train'
TEST_DATA_PATH = '/ssd003/projects/aieng/public/anomaly_detection_datasets/UCSD_Anomaly_Dataset/UCSDped2/Test'

config_path = 'Dataset/data_config.yaml'
with open(config_path) as cf_file:
    data_config = yaml.safe_load(cf_file.read())['data_config']

config_path = 'DMAD_PDM/model_config.yaml'
with open(config_path) as cf_file:
    model_config = yaml.safe_load(cf_file.read())['experiment']

train_dataset = load_train_data(TRAIN_DATA_PATH,data_config)
test_datasets, test_ground_truth = load_test_data(TEST_DATA_PATH,data_config,inference_video_name=None)

model = DMAD(model_config, train_dataset, test_datasets, test_ground_truth)
model.load_model(LOAD_PATH+'ped2.pth')

score = model.predict_score()

# Compute the AUC
accuracy = roc_auc_score(y_true=1 - test_ground_truth, y_score=score)

print('The result of evaluation')
print('AUC: ', accuracy*100)