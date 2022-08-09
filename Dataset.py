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

# For the race4
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
    """
    This class is for the Dataset where takes the inputs, the labels and the sensitive features, abstract
     all the function from the torch.utils.data.Dataset
    """
    def __init__(self, X, y, classes, is_transform=False, use_sf='race4'):
        """
        This function is the initialization of the dataset
        :param X: list, a list with all the image path inputs for the model
        :param y: list, a list with all the labels and the sensitive feature for each image input
        :param classes: list, a list with al the classes
        :param is_transform: bool, a boolean where said if the X are transformer or not
        :param use_sf: str, the name od the sensitive feature label
        """
        # The directory of the dataset
        self.dir = dataset_dir
        # self.emotion_to_id = {'Angry': 0, 'Disgust': 1, 'Fear': 2, 'Happy': 3, 'Neutral': 4, 'Sad': 5, 'Surprise': 6}
        # Get all the annotation labels
        labels, race_labels, race_4_labels, gender_labels, group_age_labels = zip(*y)
        # Get the emotion classes
        self.emotions_classes = classes
        self.n_classes = len(self.emotions_classes)
        # set all the inputs images paths
        self.image_path_x = X

        self.image_label_y = list(labels)
        self.race_labels = list(race_labels)
        self.race_4_labels = list(race_4_labels)
        self.gender_labels = list(gender_labels)
        self.age_group_labels = list(group_age_labels)

        # said what sensitive feature group to set
        self.use_sf = use_sf
        self.is_transform = is_transform

        group_array = []
        y_array = []
        if use_sf == 'race4':
            # Get the sensitive features labels groups for the group-dro for the four races
            self.n_groups = len(races4)
            for x, y, g in self:
                group_array.append(races4_to_id[g])
                y_array.append(y)
            self._group_array = torch.LongTensor(group_array)
            self._y_array = torch.LongTensor(y_array)
            self._group_counts = (torch.arange(self.n_groups).unsqueeze(1) == self._group_array).sum(1).float()

            self._y_counts = (torch.arange(len(self.emotions_classes)).unsqueeze(1) == self._y_array).sum(1).float()

        elif use_sf == 'age':
            # Get the sensitive features labels groups for the group-dro for the ages
            self.n_groups = len(binary_age_groups)
            for x, y, g in self:
                group_array.append(binary_age_groups_to_id[g])
                y_array.append(y)

            self._group_array = torch.LongTensor(group_array)
            self._y_array = torch.LongTensor(y_array)

            self._group_counts = (torch.arange(self.n_groups).unsqueeze(1) == self._group_array).sum(1).float()
            self._y_counts = (torch.arange(len(self.emotions_classes)).unsqueeze(1) == self._y_array).sum(1).float()

    def merge_data(self, X, y, random_permutation=True):
        """
        This function takes the img paths X and the labels and sensitive features y to merge to the Dataset
        :param X: list, a list with the images paths for each image
        :param y: list, a list with all the labels and the sensitive feature for each image input
        :param random_permutation: bool, A statement where said if will be random permutation on the data merge
        :return: None
        """
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

        if use_sf == 'race4':
            # Get the sensitive features labels groups for the group-dro for the four races
            self.n_groups = len(races4)
            for x, y, g in self:
                group_array.append(races4_to_id[g])
                y_array.append(y)
            self._group_array = torch.LongTensor(group_array)
            self._y_array = torch.LongTensor(y_array)
            self._group_counts = (torch.arange(self.n_groups).unsqueeze(1) == self._group_array).sum(1).float()

            self._y_counts = (torch.arange(len(self.emotions_classes)).unsqueeze(1) == self._y_array).sum(1).float()

        elif use_sf == 'age':
            # Get the sensitive features labels groups for the group-dro for the ages
            self.n_groups = len(binary_age_groups)
            for x, y, g in self:
                group_array.append(binary_age_groups_to_id[g])
                y_array.append(y)

            self._group_array = torch.LongTensor(group_array)
            self._y_array = torch.LongTensor(y_array)

            self._group_counts = (torch.arange(self.n_groups).unsqueeze(1) == self._group_array).sum(1).float()
            self._y_counts = (torch.arange(len(self.emotions_classes)).unsqueeze(1) == self._y_array).sum(1).float()

    def __len__(self):
        """
        This function return the length of the Dataset
        :return: int, the length of the dataset
        """
        return len(self.image_path_x)

    def group_counts(self):
        """
        This function return the group count of the sensitive feature for the group-DRO
        :return: torch.Tensor, a tensor with the group counts
        """
        return self._group_counts

    def class_counts(self):
        """
        This function returns the y labels counts for the group-DRO
        :return: torch.Tensor, a tensor with the y labels counts
        """
        return self._y_counts

    def input_size(self):
        """
        This function return the input size
        :return:
        """
        for x, y, g in self:
            return x.size()

    def __getitem__(self, index):
        """
        Getting the index return the data of the input image, the ground trouth label and the sensitive feature label
         on this index
        :param index: int, the index
        :return: torch.Tensor, torch.Tensor, str, The img tensor, the label tensor and the sensitive feature
        """
        image_path = self.image_path_x[index]
        label_y = self.image_label_y[index]
        img = cv2.imread(image_path)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
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
        """
        Create a data loader for the dataset and create the training, validation and testing dataset
        :param pre_processing: pre_processing, a pre_processing class with the subpopulation shift splits
        :param classes: list, a list with all the emotion classes that must have load their data
        :param use_subpopulation: bool, this statement said if have subpopulation shift or not
        :param sensitive_feature: str, this said what is sensitive feature(can be race4, age)
        """
        # define the dataset directory
        self.dir = dataset_dir
        # The classes and the emotion maps
        self.emotions_classes = classes
        self.class_map = emotion_map

        self.img_paths = []
        self.labels = []
        self.race_labels = []
        self.race_4_labels = []
        self.gender_labels = []
        self.group_age_labels = []
        self.val_img_paths = []
        self.val_labels = []
        self.val_race_labels = []
        self.val_race_4_labels = []
        self.val_gender_labels = []
        self.val_group_age_labels = []
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
        # load the validation data
        self.load_validation_data()

    def split_data_per_class(self):
        """
        This class split the data per class using the pre_processing class
        :return: None
        """
        dir_path = os.getcwd()
        path_dataset_dir = os.path.join(dir_path, self.dir)

        df_train = pd.DataFrame()
        df_test = pd.DataFrame()
        # take the data for only the emotion that must be load for training and testing
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
        # for each row from the dataframe take the image path and the labels based the key column name
        for index, row in tqdm(df_train.iterrows()):
            img_paths.append(os.path.join(path_dataset_dir, row['pth']))
            labels.append(emotion_map[row['label']])
            race_4_labels.append(row['race4'])
            race_labels.append(row['race'])
            gender_labels.append(row['gender'])
            group_age_labels.append(row['age'])

        # zip the data to can be shuffle (random permutation)
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

        # create the training Dataset based the image paths and the
        # labels of ground truth and the demographic group annotations
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
        # for each row from the dataframe take the image path and the labels based the key column name
        for index, row in tqdm(df_test.iterrows()):
            img_paths.append(os.path.join(path_dataset_dir, row['pth']))
            labels.append(emotion_map[row['label']])
            race_4_labels.append(row['race4'])
            race_labels.append(row['race'])
            gender_labels.append(row['gender'])
            group_age_labels.append(row['age'])

        # make a random permutation on the data
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
        # create the testing Dataset based the image paths and the
        # labels of ground truth and the demographic group annotations
        self.testing_dataset = Dataset(X=img_paths,
                                       y=labels_with_sens_features,
                                       is_transform=False,
                                       use_sf=self.sensitive_feature,
                                       classes=self.emotions_classes)

    def load_data(self):
        """
        This function is to load the data from the dataset csv and get only the data from the emotion classes that
        wants to be load from the data loader
        :return: None
        """
        dir_path = os.getcwd()
        path_dataset_dir = os.path.join(dir_path, self.dir)
        path_csv_sens_features = os.path.join(path_dataset_dir, race_labels_filename)
        path_csv_labels = os.path.join(path_dataset_dir, data_filename)

        # Load the labels and the sensitive features from the csv
        csv_sf = pd.read_csv(path_csv_sens_features)
        csv_labels = pd.read_csv(path_csv_labels)

        # Merge the two dataframes based the image path
        data = pd.merge(csv_sf, csv_labels, on="pth")
        for index, row in tqdm(data.iterrows()):
            image_path = row['pth']
            label = row['label']
            race_4 = row['race4']
            race = row['race']
            gender = row['gender']
            age_group = row['age']

            # keep only the classes that was defined on the initialization of the Class
            if label in self.emotions_classes:
                self.img_paths.append(os.path.join(path_dataset_dir, image_path))
                self.labels.append(emotion_map[row['label']])
                self.race_4_labels.append(race_4)
                self.race_labels.append(race)
                self.gender_labels.append(gender)
                self.group_age_labels.append(age_group)

    def load_validation_data(self):
        """
        Based on the validation csv data load the data for validation only for the emotion classes that
        wants to be load from the data loader
        :return: None
        """
        dir_path = os.getcwd()
        path_dataset_dir = os.path.join(dir_path, self.dir)
        path_csv_sens_features = os.path.join(path_dataset_dir, val_race_labels_filename)
        path_csv_labels = os.path.join(path_dataset_dir, val_data_filename)

        # Load the csv data into the dataframe
        csv_sf = pd.read_csv(path_csv_sens_features)
        csv_labels = pd.read_csv(path_csv_labels)

        # Merge the data based the pth name on their column
        data = pd.merge(csv_sf, csv_labels, on="pth")
        for index, row in tqdm(data.iterrows()):
            image_path = row['pth']
            label = row['label']
            race_4 = row['race4']
            race = row['race']
            gender = row['gender']
            age_group = row['age']

            # keep only the classes that was defined on the initialization of the Class
            if label in self.emotions_classes:
                self.val_img_paths.append(os.path.join(path_dataset_dir, image_path))
                self.val_labels.append(emotion_map[row['label']])
                self.val_race_4_labels.append(race_4)
                self.val_race_labels.append(race)
                self.val_gender_labels.append(gender)
                self.val_group_age_labels.append(age_group)

        labels_with_sens_features = zip(self.val_labels, self.val_race_labels,
                                        self.val_race_4_labels, self.val_gender_labels, self.val_group_age_labels)
        labels_with_sens_features = list(labels_with_sens_features)
        # Create the validation data
        self.validation_dataset = Dataset(X=self.val_img_paths,
                                          y=labels_with_sens_features,
                                          is_transform=False,
                                          classes=self.emotions_classes,
                                          use_sf=self.sensitive_feature)

    def split_data(self, train=0.7):
        """
        Split the data to testing and training class
        :param train: float, the percentage of the training data
        :return:
        """
        labels_with_sens_features = zip(self.labels, self.race_labels, self.race_4_labels, self.gender_labels,
                                        self.group_age_labels)
        labels_with_sens_features = list(labels_with_sens_features)

        train_x, test_x, train_y, test_y = train_test_split(self.img_paths,
                                                            labels_with_sens_features,
                                                            train_size=train,
                                                            random_state=0)

        self.training_dataset = Dataset(X=train_x, y=train_y,
                                        is_transform=False, classes=self.emotions_classes,
                                        use_sf=self.sensitive_feature)
        self.testing_dataset = Dataset(X=test_x, y=test_y,
                                       is_transform=False, classes=self.emotions_classes,
                                       use_sf=self.sensitive_feature)

    def merge_training_data(self, emotion_classes, datasets, percentage=0.2):
        """
        Getting a dataset and his emotion classes merge with the dataset with only the percentage of the data
        :param emotion_classes: list, a list from the emotion classes that must merge with the dataset
        :param datasets: list, a list with all the dataset class
        :param percentage: float, the percentage of the data that must keep
        :return: None
        """
        dir_path = os.getcwd()
        path_dataset_dir = os.path.join(dir_path, self.dir)

        # if used subpopulation get the data from the preprocessing
        if self.use_subpopulation:
            df_train = pd.DataFrame()
            # for emotion class get the same amount of percentage on the merge of data
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
            # other wise get the data from the dataset list
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
