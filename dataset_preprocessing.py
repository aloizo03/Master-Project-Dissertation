import os
from config import *
import pandas as pd
from sklearn.model_selection import train_test_split

races4 = {'Asian', 'Black', 'Indian', 'White'}
non_white_races4 = {'Black', 'Indian'}
white_races4 = {'Asian', 'White'}


class pre_processing:

    def __init__(self, train=0.7, test=0.3, sensitive_features='race4'):
        dir_path = os.getcwd()
        path_dataset_dir = os.path.join(dir_path, dataset_dir)
        path_csv_sens_features = os.path.join(path_dataset_dir, race_labels_filename)
        path_csv_labels = os.path.join(path_dataset_dir, data_filename)

        self.csv_sf = pd.read_csv(path_csv_sens_features)
        self.csv_labels = pd.read_csv(path_csv_labels)
        self.data = pd.merge(self.csv_sf, self.csv_labels, on="pth")

        self.test = test
        self.train = train
        self.sensitive_feature = sensitive_features

        # Have training data per sensitive feature
        self.training_data = {}
        self.testing_data = {}
        self.split_per_class_and_per_sens_feature()

    def split_for_all_classes(self):
        if self.sensitive_feature == 'race4':
            for race in races4:
                df = self.data.loc[self.data['race4'] == race]
                train_df, test_df = train_test_split(df, test_size=self.test)
                self.training_data[race] = train_df
                self.testing_data[race] = test_df

    def split_per_class_and_per_sens_feature(self):
        for emotion in emotions_classes:
            df_emotion = self.data.loc[self.data['label'] == emotion]
            if self.sensitive_feature == 'race4':
                for race in races4:
                    df = df_emotion.loc[df_emotion['race4'] == race]
                    train_df, test_df = train_test_split(df, test_size=self.test)
                    self.training_data[race] = pd.concat([self.training_data.get(race, pd.DataFrame()), train_df])
                    self.testing_data[race] = pd.concat([self.testing_data.get(race, pd.DataFrame()), test_df])

    def get_testing_per_class(self, emotion_class):
        df_test = pd.DataFrame()
        for race, data in self.testing_data.items():
            df_test = pd.concat([df_test, data.loc[self.data['label'] == emotion_class]])
        return df_test

    def get_training_per_class(self, emotion_class):
        df_train = pd.DataFrame()
        for race, data in self.training_data.items():
            df_train = pd.concat([df_train, data.loc[self.data['label'] == emotion_class]])
        return df_train

    def get_testing_per_sensitive_feature(self, sensitive_feature):

        return self.testing_data[sensitive_feature]

    def get_training_per_sensitive_feature(self, sensitive_feature):

        return self.training_data[sensitive_feature]
