model_input = (224, 224)
colour_channel_input = 3
dataset_dir = 'Dataset/input/affectnetsample/'
data_filename = 'labels.csv'
race_labels_filename = 'test_outputs.csv'
model_dir = 'models/'

emotions_classes = ['anger', 'contempt', 'disgust', 'fear', 'sad', 'surprise', 'happy', 'neutral']
# emotion_map = {0: 'anger', 1: 'contempt', 2: 'disgust', 3: 'fear', 4: 'happy', 5: 'neutral', 6: 'sad', 7: 'surprise'}
emotion_map = {'anger': 0, 'contempt': 1, 'disgust': 2, 'fear': 3, 'sad': 4, 'surprise': 5, 'happy': 6, 'neutral': 7}

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
