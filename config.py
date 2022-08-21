# CNN image input RGB colour
model_input = (224, 224)
colour_channel_input = 3
# The dataset direction on the computer the path start from the project directory
dataset_dir = 'Dataset/input/affectnet/'
# filename of the training labels and sensitive features annotations
# data_filename = 'labels.csv'
# race_labels_filename = 'test_outputs.csv'
# This filenames is for balance dataset
data_filename = 'balance_dataset_labels.csv'
race_labels_filename = 'balance_dataset_annotations.csv'
# filename for the validation labels and sensitive features annotations
val_data_filename = 'val_labels.csv'
val_race_labels_filename = 'val_test_outputs.csv'
# where will be saved the emotion classes
model_dir = 'models/'

# emotions_classes = ['anger', 'contempt', 'disgust', 'fear', 'sad', 'surprise', 'happy', 'neutral']
emotions_classes = ['anger', 'sad', 'disgust', 'fear', 'contempt', 'surprise', 'happy', 'neutral']
emotion_map = {'anger': 0, 'sad': 1, 'disgust': 2, 'fear': 3, 'contempt': 4, 'surprise': 5, 'happy': 6, 'neutral': 7}

races4_to_id = {'Asian': 0, 'Black': 1, 'Indian': 2, 'White': 3}
binary_age_groups_to_id = {'young': 0, 'old': 1}

# The VGG architecture to can be initialize on the Model class ( This must not be changed)
# M is for Max-pool Layer
VGG_19_model_architecture = [64,
                             64,
                             "M",
                             128,
                             128,
                             "M",
                             256,
                             256,
                             256,
                             256,
                             "M",
                             512,
                             512,
                             512,
                             512,
                             "M",
                             512,
                             512,
                             512,
                             512,
                             "M", ]
# For group DRO values
generalization_adjustment = '0'

robust_step_size = 0.01

use_normalized_loss = False

btl = False

minimum_variational_weight = 0

is_robust = True

alpha = None

gamma = 0.1
