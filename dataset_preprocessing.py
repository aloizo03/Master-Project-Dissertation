import os
from config import *
import pandas as pd
from sklearn.model_selection import train_test_split

# for races
races4 = {'Asian', 'Black', 'Indian', 'White'}
non_white_races4 = {'Black', 'Indian'}
white_races4 = {'Asian', 'White'}

# for age group
age_groups = {'0-2', '10-19', '20-29', '3-9', '30-39', '40-49', '50-59', '60-69', '70+'}
young_age_group = {'0-2', '3-9', '10-19', '20-29'}
old_age_group = {'30-39', '40-49', '50-59', '60-69', '70+'}
binary_age_groups = {'young', 'old'}


class pre_processing:
    """
    This class is for a pre-processing for the dataset before the training and the evaluation.
    """
    def __init__(self, train=0.7, test=0.3, sensitive_features='race4'):
        """
        This class is for the initialization of the class
        :param train: float, the percentage of the training data
        :param test: float, the percentage of the testing data
        :param sensitive_features: str, the name of the sensitive feature
        """
        # The directory of the csv files of the datasets and load them as dataframe from pandas
        dir_path = os.getcwd()
        path_dataset_dir = os.path.join(dir_path, dataset_dir)
        path_csv_sens_features = os.path.join(path_dataset_dir, race_labels_filename)
        path_csv_labels = os.path.join(path_dataset_dir, data_filename)

        self.csv_sf = pd.read_csv(path_csv_sens_features)
        self.csv_labels = pd.read_csv(path_csv_labels)
        # merge the two csv files based of the filename
        self.data = pd.merge(self.csv_sf, self.csv_labels, on="pth")

        self.test = test
        self.train = train
        self.sensitive_feature = sensitive_features

        # Have training data per sensitive feature into a dictionary
        self.training_data = {}
        self.testing_data = {}
        self.filter_sensitive_feature()
        self.split_per_class_and_per_sens_feature()

    def filter_sensitive_feature(self):
        """
        This function change the sensitive feature of age into a binary of young and old
        :return: None
        """
        # make the age sensitive feature to binary
        if self.sensitive_feature == 'age':
            for age_group in young_age_group:
                self.data.loc[self.data['age'] == age_group, 'age'] = 'young'

            for age_group in old_age_group:
                self.data.loc[self.data['age'] == age_group, 'age'] = 'old'

    def split_for_all_classes(self):
        """
        For all the classes spit the data based the sensitive feature into equal percentage of each of them
        will have same percentage of data and for training and for evaluation. The same amount cannot be because the
        dataset is unbalance in terms of sensitive features
        :return:
        """
        if self.sensitive_feature == 'race4':
            # Make a fair split per sensitive feature to have the same percentage of data and for training and testing
            for race in races4:
                # take the data were have the specific sensitive feature
                df = self.data.loc[self.data['race4'] == race]
                train_df, test_df = train_test_split(df, test_size=self.test)
                # add for the race the data for training and testing
                self.training_data[race] = train_df
                self.testing_data[race] = test_df

        elif self.sensitive_feature == 'age':
            for age_group in binary_age_groups:
                # take the data were have the specific sensitive feature
                df = self.data.loc[self.data['age'] == age_group]
                train_df, test_df = train_test_split(df, test_size=self.test)
                # add for the race the data for training and testing
                self.training_data[age_group] = train_df
                self.testing_data[age_group] = test_df

    def split_per_class_and_per_sens_feature(self):
        """
        For each class split the same amount of percentage per sensitive feature to have the same percentage
         of data and for training and for evaluation
        :return: None
        """
        # For each class
        for emotion in emotions_classes:
            # Get the emotion for this class
            df_emotion = self.data.loc[self.data['label'] == emotion]
            # Make a fair split per sensitive feature to have the same percentage of data and for training and testing
            if self.sensitive_feature == 'race4':
                for race in races4:
                    df = df_emotion.loc[df_emotion['race4'] == race]
                    train_df, test_df = train_test_split(df, test_size=self.test)
                    self.training_data[race] = pd.concat([self.training_data.get(race, pd.DataFrame()), train_df])
                    self.testing_data[race] = pd.concat([self.testing_data.get(race, pd.DataFrame()), test_df])
            elif self.sensitive_feature == 'age':
                for age_group in binary_age_groups:
                    df = df_emotion.loc[df_emotion['age'] == age_group]
                    train_df, test_df = train_test_split(df, test_size=self.test)
                    self.training_data[age_group] = pd.concat(
                        [self.training_data.get(age_group, pd.DataFrame()), train_df])
                    self.testing_data[age_group] = pd.concat(
                        [self.testing_data.get(age_group, pd.DataFrame()), test_df])

    def get_testing_per_class(self, emotion_class):
        """
        This class getting the emotion class return all his data which split for testing
        :param emotion_class: str, the name of the class
        :return: DataFrame, a dataframe of all the testing data of the given emotion class
        """
        df_test = pd.DataFrame()
        for sf, data in self.testing_data.items():
            df_test = pd.concat([df_test, data.loc[self.data['label'] == emotion_class]])
        return df_test

    def get_training_per_class(self, emotion_class):
        """
        This class getting the emotion class return all his data which split for training
        :param emotion_class: str, the name of the class
        :return: DataFrame, a dataframe of all the training data of the given emotion class
        """
        df_train = pd.DataFrame()
        for sf, data in self.training_data.items():
            df_train = pd.concat([df_train, data.loc[self.data['label'] == emotion_class]])
        return df_train

    def get_testing_per_sensitive_feature(self, sensitive_feature):
        """
        Getting the sensitive feature name return the testing data for the specific sensitive feature group
        :param sensitive_feature: str, the name of the sensitive feature
        :return: DataFrame, a dataframe of all the testing data of the given sensitive feature group
        """
        return self.testing_data[sensitive_feature]

    def get_training_per_sensitive_feature(self, sensitive_feature):
        """
        Getting the sensitive feature name return the training data for the specific sensitive feature group
        :param sensitive_feature: str, the name of the sensitive feature
        :return: DataFrame, a dataframe of all the training data of the given sensitive feature group
        """
        return self.training_data[sensitive_feature]
