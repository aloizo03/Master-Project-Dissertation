import os.path

import pandas as pd
import torch
from torch.utils import data
from PIL import Image
import numpy as np
import cv2
from tqdm import tqdm
from config import *
from sklearn.model_selection import train_test_split
from sklearn.model_selection import ShuffleSplit
import random
from transform import *

races4 = {'Asian', 'Black', 'Indian', 'White'}
races4_to_id = {'Asian': 0, 'Black': 1, 'Indian': 2, 'White': 3}
non_white_races4 = {'Black', 'Indian'}
white_races4 = {'Asian', 'White'}

# for age group
age_groups = {'0-2', '10-19', '20-29', '3-9', '30-39', '40-49', '50-59', '60-69', '70+'}
young_age_group = {'0-2', '3-9', '10-19', '20-29'}
old_age_group = {'30-39', '40-49', '50-59', '60-69', '70+'}
binary_age_groups = {'young', 'old'}
binary_age_groups_to_id = {'young': 0, 'old': 1}


class Dataset(data.Dataset):

    def __init__(self, X, y, classes, is_transform=False, use_sf='race4'):
        self.dir = dataset_dir
        # self.emotion_to_id = {'Angry': 0, 'Disgust': 1, 'Fear': 2, 'Happy': 3, 'Neutral': 4, 'Sad': 5, 'Surprise': 6}
        labels, race_labels, race_4_labels, gender_labels, group_age_labels = zip(*y)
        self.emotions_classes = classes
        self.n_classes = len(self.emotions_classes)
        self.image_path_x = X

        self.image_label_y = list(labels)
        self.race_labels = list(race_labels)
        self.race_4_labels = list(race_4_labels)
        self.gender_labels = list(gender_labels)
        self.age_group_labels = list(group_age_labels)

        self.use_sf = use_sf
        self.is_transform = is_transform

        group_array = []
        y_array = []
        if use_sf == 'race4':
            self.n_groups = len(races4)
            for x, y, g in self:
                group_array.append(races4_to_id[g])
                y_array.append(y)

            self._group_array = torch.LongTensor(group_array)
            self._y_array = torch.LongTensor(y_array)
            self._group_counts = (torch.arange(self.n_groups).unsqueeze(1) == self._group_array).sum(1).float()

            self._y_counts = (torch.arange(len(self.emotions_classes)).unsqueeze(1) == self._y_array).sum(1).float()

        elif use_sf == 'age':
            self.n_groups = len(binary_age_groups)
            for x, y, g in self:
                group_array.append(binary_age_groups[g])
                y_array.append(y)

            self._group_array = torch.LongTensor(group_array)
            self._y_array = torch.LongTensor(y_array)

            self._group_counts = (torch.arange(self.n_groups).unsqueeze(1) == self._group_array).sum(1).float()
            self._y_counts = (torch.arange(len(self.emotions_classes)).unsqueeze(1) == self._y_array).sum(1).float()

    def merge_data(self, X, y, random_permutation=True):
        labels, race_labels, race_4_labels, gender_labels, group_age_labels = zip(*y)

        self.image_path_x.extend(X)
        self.image_label_y.extend(labels)
        self.race_labels.extend(race_labels)
        self.race_4_labels.extend(race_4_labels)
        self.gender_labels.extend(gender_labels)
        self.age_group_labels.extend(group_age_labels)

        if random_permutation:
            zipped_data = list(
                zip(self.image_path_x, self.image_label_y, self.race_labels, self.race_4_labels, self.gender_labels,
                    self.age_group_labels))
            random.shuffle(zipped_data)
            self.image_path_x, self.image_label_y, self.race_labels, self.race_4_labels, self.gender_labels, self.age_group_labels = zip(
                *zipped_data)

    def __len__(self):
        return len(self.image_path_x)

    def group_counts(self):
        return self._group_counts

    def class_counts(self):
        return self._y_counts

    def input_size(self):
        for x, y, g in self:
            return x.size()

    def __getitem__(self, index):
        image_path = self.image_path_x[index]
        label_y = self.image_label_y[index]
        img = cv2.imread(image_path)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # img_alligned = alligned_face(img_rgb)
        if not self.is_transform:
            img_rgb = transform(image=img_rgb, size=model_input)
        img_rgb = torch.from_numpy(img_rgb).float()
        label_y = torch.tensor(label_y, dtype=torch.float)
        label_y = label_y.type(torch.LongTensor)

        if self.use_sf == 'race4':
            sensitive_feature = self.race_4_labels[index]
        elif self.use_sf == 'age':
            sensitive_feature = self.age_group_labels[index]

        return img_rgb, label_y, sensitive_feature


class DataLoader_Affect_Net:

    def __init__(self, pre_processing,
                 classes=['anger', 'contempt', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise'],
                 use_subpopulation=False, sensitive_feature='race4'):
        self.dir = dataset_dir
        self.emotions_classes = classes
        self.class_map = emotion_map
        self.img_paths = []
        self.labels = []
        self.race_labels = []
        self.race_4_labels = []
        self.gender_labels = []
        self.group_age_labels = []
        self.training_dataset = None
        self.testing_dataset = None
        self.validation_dataset = None
        self.use_subpopulation = use_subpopulation
        self.pre_processing = pre_processing
        self.sensitive_feature = sensitive_feature

        # load ans split data to Training and testing dataset
        if use_subpopulation:
            self.split_data_per_class()
        else:
            self.load_data()
            self.split_data()

    def split_data_per_class(self):
        dir_path = os.getcwd()
        path_dataset_dir = os.path.join(dir_path, self.dir)

        df_train = pd.DataFrame()
        df_test = pd.DataFrame()
        for emotion in self.emotions_classes:
            df = self.pre_processing.get_training_per_class(emotion_class=emotion)
            df_train = pd.concat([df_train, df])

            df = self.pre_processing.get_testing_per_class(emotion_class=emotion)
            df_test = pd.concat([df_test, df])

        img_paths = []
        labels = []
        race_4_labels = []
        race_labels = []
        gender_labels = []
        group_age_labels = []
        for index, row in tqdm(df_train.iterrows()):
            img_paths.append(os.path.join(path_dataset_dir, row['pth']))
            labels.append(emotion_map[row['label']])
            race_4_labels.append(row['race4'])
            race_labels.append(row['race'])
            gender_labels.append(row['gender'])
            group_age_labels.append(row['age'])
        zipped_data = list(zip(img_paths, labels, race_4_labels, race_labels, gender_labels, group_age_labels))
        random.shuffle(zipped_data)
        img_paths, labels, race_4_labels, race_labels, gender_labels, group_age_labels = zip(*zipped_data)
        img_paths = list(img_paths)

        self.img_paths.extend(img_paths)
        self.labels.extend(labels)
        self.race_4_labels.extend(race_4_labels)
        self.race_labels.extend(race_labels)
        self.gender_labels.extend(gender_labels)
        self.group_age_labels.extend(group_age_labels)
        labels_with_sens_features = zip(labels, race_labels, race_4_labels, gender_labels, group_age_labels)
        labels_with_sens_features = list(labels_with_sens_features)

        self.training_dataset = Dataset(X=img_paths,
                                        y=labels_with_sens_features,
                                        is_transform=False,
                                        classes=self.emotions_classes,
                                        use_sf=self.sensitive_feature)

        img_paths = []
        labels = []
        race_4_labels = []
        race_labels = []
        gender_labels = []
        group_age_labels = []
        for index, row in tqdm(df_test.iterrows()):
            img_paths.append(os.path.join(path_dataset_dir, row['pth']))
            labels.append(emotion_map[row['label']])
            race_4_labels.append(row['race4'])
            race_labels.append(row['race'])
            gender_labels.append(row['gender'])
            group_age_labels.append(row['age'])

        zipped_data = list(zip(img_paths, labels, race_4_labels, race_labels, gender_labels, group_age_labels))
        random.shuffle(zipped_data)
        img_paths, labels, race_4_labels, race_labels, gender_labels, group_age_labels = zip(*zipped_data)
        img_paths = list(img_paths)

        self.img_paths.extend(img_paths)
        self.labels.extend(labels)
        self.race_4_labels.extend(race_4_labels)
        self.race_labels.extend(race_labels)
        self.gender_labels.extend(gender_labels)
        self.group_age_labels.extend(group_age_labels)
        labels_with_sens_features = zip(labels, race_labels, race_4_labels, gender_labels, group_age_labels)
        labels_with_sens_features = list(labels_with_sens_features)
        self.testing_dataset = Dataset(X=img_paths,
                                       y=labels_with_sens_features,
                                       is_transform=False,
                                       use_sf=self.sensitive_feature,
                                       classes=self.emotions_classes)

    def load_data(self):
        dir_path = os.getcwd()
        path_dataset_dir = os.path.join(dir_path, self.dir)
        path_csv_sens_features = os.path.join(path_dataset_dir, race_labels_filename)
        path_csv_labels = os.path.join(path_dataset_dir, data_filename)

        csv_sf = pd.read_csv(path_csv_sens_features)
        csv_labels = pd.read_csv(path_csv_labels)

        data = pd.merge(csv_sf, csv_labels, on="pth")
        for index, row in tqdm(data.iterrows()):
            image_path = row['pth']
            label = row['label']
            race_4 = row['race4']
            race = row['race']
            gender = row['gender']
            age_group = row['age']

            # return only the classes that the User select
            if label in self.emotions_classes:
                self.img_paths.append(os.path.join(path_dataset_dir, image_path))
                self.labels.append(emotion_map[row['label']])
                self.race_4_labels.append(race_4)
                self.race_labels.append(race)
                self.gender_labels.append(gender)
                self.group_age_labels.append(age_group)

    def split_data(self, train=0.7):
        labels_with_sens_features = zip(self.labels, self.race_labels, self.race_4_labels, self.gender_labels,
                                        self.group_age_labels)
        labels_with_sens_features = list(labels_with_sens_features)

        train_x, test_x, train_y, test_y = train_test_split(self.img_paths,
                                                            labels_with_sens_features,
                                                            train_size=train,
                                                            random_state=0)

        self.training_dataset = Dataset(X=train_x, y=train_y, is_transform=False, classes=self.emotions_classes)
        self.testing_dataset = Dataset(X=test_x, y=test_y, is_transform=False, classes=self.emotions_classes)

    def merge_training_data(self, emotion_classes, datasets, percentage=0.3):
        dir_path = os.getcwd()
        path_dataset_dir = os.path.join(dir_path, self.dir)

        if self.use_subpopulation:
            df_train = pd.DataFrame()
            for emotion in emotion_classes:
                df = self.pre_processing.get_training_per_class(emotion_class=emotion)
                df = df.sample(frac=percentage)
                df_train = pd.concat([df_train, df])
            img_paths = []
            labels = []
            race_4_labels = []
            race_labels = []
            gender_labels = []
            group_age_labels = []
            for index, row in df_train.iterrows():
                img_paths.append(os.path.join(path_dataset_dir, row['pth']))
                labels.append(emotion_map[row['label']])
                race_4_labels.append(row['race4'])
                race_labels.append(row['race'])
                gender_labels.append(row['gender'])
                group_age_labels.append(row['age'])

            zipped_data = list(zip(img_paths, labels, race_4_labels, race_labels, gender_labels, group_age_labels))
            random.shuffle(zipped_data)

            img_paths, labels, race_4_labels, race_labels, gender_labels, group_age_labels = zip(*zipped_data)
            labels_with_sens_features = zip(labels, race_labels, race_4_labels, gender_labels, group_age_labels)
            labels_with_sens_features = list(labels_with_sens_features)
            img_paths = list(img_paths)

            self.training_dataset.merge_data(X=img_paths, y=labels_with_sens_features, random_permutation=True)
        else:
            img_paths = []
            labels = []
            race_4_labels = []
            race_labels = []
            gender_labels = []
            group_age_labels = []
            for dataset in datasets:
                img_paths.extend(random.sample(dataset.img_paths, len(dataset.img_paths) * percentage))
                labels.extend(random.sample(dataset.labels, len(dataset.labels) * percentage))
                race_4_labels.extend(random.sample(dataset.race_4_labels, len(dataset.race_4_labels) * percentage))
                race_labels.extend(random.sample(dataset.race_labels, len(dataset.race_labels) * percentage))
                gender_labels.extend(random.sample(dataset.gender_labels, len(dataset.gender_labels) * percentage))
                group_age_labels.extend(
                    random.sample(dataset.age_group_labels, len(dataset.age_group_labels) * percentage))

            zipped_data = list(zip(img_paths, labels, race_4_labels, race_labels, gender_labels, group_age_labels))
            random.shuffle(zipped_data)

            img_paths, labels, race_4_labels, race_labels, gender_labels, group_age_labels = zip(*zipped_data)
            labels_with_sens_features = zip(labels, race_labels, race_4_labels, gender_labels, group_age_labels)
            labels_with_sens_features = list(labels_with_sens_features)
            img_paths = list(img_paths)

            self.training_dataset.merge_data(X=img_paths, y=labels_with_sens_features, random_permutation=True)

    # def concat_datasets(self, datasets):
    #     train_x = []
    #     train_y = []
    #     test_x = []
    #     test_y = []
    #     classes = []
    #     self.img_paths = []
    #     self.labels = []
    #     for dataset in datasets:
    #         train_x.extend(dataset.training_dataset.image_path_x)
    #         train_y.extend(dataset.training_dataset.image_label_y)
    #         test_x.extend(dataset.testing_dataset.image_path_x)
    #         test_y.extend(dataset.testing_dataset.image_label_y)
    #         classes.extend(dataset.emotions_classes)
    #         self.img_paths.extend(dataset.img_paths)
    #         self.labels.extend(dataset.labels)
    #
    #     # train_x, train_y = shuffle(np.array(train_x), np.array(train_y))
    #     # test_x, test_y = shuffle(np.array(test_x), np.array(test_y))
    #
    #     self.emotions_classes = classes
    #     self.training_dataset = Dataset(X=train_x, y=train_y, is_transform=False, classes=self.emotions_classes)
    #     self.testing_dataset = Dataset(X=test_x, y=test_y, is_transform=False, classes=self.emotions_classes)
