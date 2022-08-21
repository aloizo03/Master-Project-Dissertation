import copy
import os

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from tqdm import tqdm
from collections import OrderedDict
import torchvision
from Dataset import DataLoader_Affect_Net
from torch.utils import data
from config import *
from torch.autograd import Variable
import torch.nn.functional as F
from loss import LossComputer


def kaiming_normal_init(model):
    if isinstance(model, nn.Conv2d):
        nn.init.kaiming_normal_(model.weight, nonlinearity='relu')
    elif isinstance(model, nn.Linear):
        nn.init.kaiming_normal_(model.weight, nonlinearity='sigmoid')


class Model(nn.Module):
    """
    This class is the model architecture of the VGG 19
    """

    def __init__(self, lr, epochs, batch_size, loss_function, num_hidden=4096, use_batch_norm=False, use_pre_trained_weights=False,
                 weight_decay=0.0005,
                 emotion_classes=['anger', 'contempt', 'disgust', 'fear', 'sad', 'surprise', 'happy', 'neutral'],
                 use_subpopulation=True, sensitive_feature='race4'):
        """
        The initialize function where the model are initialize
        :param lr: float, the learning rate of the model value
        :param epochs: int, the total number of epochs
        :param batch_size: int, the total value of the batch size
        :param num_hidden: int, the number of the hidden layer in fully connected
        :param use_batch_norm: bool, a boolean state where said if had Batch normalization the model
        :param use_pre_trained_weights: bool, A boolean state where said if must be used pre trained weights
        :param weight_decay: float, the value of weight decay for the optimizer
        :param emotion_classes: list, a list with the name of the classes
        :param use_subpopulation: bool, a boolean statement where said if use subpopulation shift with group-DRO
        :param sensitive_feature: str, the name of the sensitive feature (works only for race4 and age )
        """
        # Initialize Hyper parameters
        self.sensitive_feature = sensitive_feature
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size

        self.use_batch_norm = use_batch_norm
        self.use_pre_trained_weights = use_pre_trained_weights
        # SGD optimization values
        self.momentum = 0.9
        self.weight_decay = weight_decay
        # get the device the best is to have a GPU with the CUDA
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.use_subpopulation = use_subpopulation

        super().__init__()
        # classes information (Affect Net has 8 emotion classes)
        self.num_classes = len(emotion_classes)
        self.emotion_classes = emotion_classes

        self.input_width, self.input_height = model_input
        self.colour_channel = colour_channel_input
        self.total_hidden_layers = num_hidden
        self.model_architecture = VGG_19_model_architecture
        # Creating the model
        self.feature_extractor = self.create_feature_extractor()
        self.fc = self.create_fully_connected_layer()
        self.fc.apply(kaiming_normal_init)

        self.loss_function = loss_function
        if self.use_pre_trained_weights:
            self.add_pre_trained_weights()
        else:
            self.feature_extractor.apply(kaiming_normal_init)

    def classify(self, images):
        """
        Taking multiple images classify for each image their emotion
        :param images: torch.Tensor, a torch tensor with all the images
        :return: list: a list with all the predictions
        """
        _, preds = torch.max(torch.softmax(self.forward(images), dim=1), dim=1, keepdim=False)

        return preds

    def add_pre_trained_weights(self):
        """
        Add pretrained weights on the feature extractor of the model, if has batch normalization
         add the batch normilization pre trained weights
        :return: None
        """
        if self.use_batch_norm:
            # Get vgg with batch normalization
            vgg_19 = torchvision.models.vgg19_bn(pretrained=True, progress=False)
        else:
            # Get original version of vgg-19
            vgg_19 = torchvision.models.vgg19(pretrained=True, progress=False)
        for param1, param2 in zip(vgg_19.features, self.feature_extractor):
            # Change the weights only in Convolution layers and Batch normalization
            if isinstance(param1, nn.Conv2d) or isinstance(param1, nn.BatchNorm2d):
                param2.weight.data = param1.weight.data
                param2.bias.data = param1.bias.data

        self.feature_extractor = self.feature_extractor.to(self.device)

    def calculate_accuracy(self, predictions, gt_labels):
        """
        Make predictions and calculate the accuracy
        :param predictions: torch.Tensor, a tensor with the model output predictions
        :param gt_labels:  torch.Tensor, a tensor with the ground ruth label for each prediction
        :return: float: the accuracy of the model prediction for this predictions
        """
        _, predicts = torch.max(torch.softmax(predictions, dim=1), dim=1, keepdim=False)
        corrects = (predicts == gt_labels).sum()
        corrects = corrects.cpu().detach().numpy()

        return corrects / gt_labels.size(0)

    def forward(self, img_x):
        """
        Getting an image make the model prediction
        :param img_x: torch.Tensor, A tensor with images
        :return: torch.Tensor: A tensor with the prediction for each image
        """
        # The result from the Convolutional Layers
        x = self.feature_extractor(img_x)
        x = x.view(x.size(0), -1)
        # Pass the result from the fully connected layer
        out = self.fc(x)
        return out

    def create_feature_extractor(self):
        """
        This function create the feature extractor and return it
        :return: nn.Sequential: a Sequential with all the feature extractor layers of the network
        """
        layers = []
        input_channels = self.colour_channel

        for layer in self.model_architecture:
            if type(layer) == int:
                out_channel = layer
                # If have BN add and a batch normalization to the layer
                if self.use_batch_norm:
                    layers.extend([
                        nn.Conv2d(in_channels=input_channels, out_channels=out_channel, kernel_size=(3, 3),
                                  stride=(1, 1),
                                  padding=(1, 1)),
                        nn.BatchNorm2d(out_channel),
                        nn.ReLU(),
                    ])
                    input_channels = layer
                else:
                    layers.extend([
                        nn.Conv2d(in_channels=input_channels, out_channels=out_channel, kernel_size=(3, 3),
                                  stride=(1, 1),
                                  padding=(1, 1)),
                        nn.ReLU(),
                    ])
                    input_channels = layer
            else:
                layers.append(nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)))

        # Put all the layers to a Sequential
        layers_Features_extraction = nn.Sequential(*layers).to(self.device)
        return layers_Features_extraction

    def create_fully_connected_layer(self):
        """
        This function create the FC layer and return it
        :return: nn.Sequential: A Sequential with all the FC layers
        """

        total_max_pool = self.model_architecture.count("M")
        factor = 2 ** total_max_pool

        if self.input_width % factor != 0 and self.input_height % factor != 0:
            raise ValueError(f"`in_height` and `in_width` must be multiples of {factor}")
        output_height = self.input_height // factor
        output_width = self.input_width // factor
        last_output_channel = next(
            layer_size for layer_size in self.model_architecture[::-1] if type(layer_size) == int)
        # Calculate the input size of the fc layer by the last output layer size
        self.fc_input_size = last_output_channel * output_height * output_width

        fc_layer = nn.Sequential(OrderedDict([
            ('fc1', nn.Linear(self.fc_input_size, self.total_hidden_layers)),
            ('relu1', nn.ReLU()),
            ('dropout1', nn.Dropout(p=0.5)),
            ('fc2', nn.Linear(self.total_hidden_layers, self.total_hidden_layers)),
            ('relu2', nn.ReLU()),
            ('dropout2', nn.Dropout(p=0.5)),
            ('output', nn.Linear(self.total_hidden_layers, self.num_classes)),
        ])).to(self.device)
        return fc_layer

    def training_(self, dataset):
        """
        Training the model of the original emotion classes
        :param dataset: Dataset, a dataset class with the training, validation and testing dataset
        :return:
        """
        training_dict = {}
        total_loss = []
        accuracy_lst = []

        train_data_loader = data.DataLoader(dataset=dataset.training_dataset, batch_size=self.batch_size, shuffle=False,
                                            num_workers=0)

        # Observe that all parameters are being optimized
        optimizer = torch.optim.SGD(self.parameters(), lr=self.lr, momentum=self.momentum,
                                    weight_decay=self.weight_decay)
        with tqdm(total=self.epochs) as pbar:
            epoch_dict = {}
            for epoch in range(self.epochs):
                running_loss = 0.0
                accuracy = 0
                count = 0
                predicts_lst = []
                labels_lst = []
                sensitive_feature_labels_lst = []
                for step, datas in enumerate(train_data_loader):
                    images, labels, sf = datas
                    images = images.to(self.device)
                    labels = labels.to(self.device)

                    # make the sf from str to id
                    if self.sensitive_feature == 'race4':
                        sf_to_id = [races4_to_id[i] for i in sf]
                    elif self.sensitive_feature == 'age':
                        sf_to_id = [binary_age_groups_to_id[i] for i in sf]

                    # zero the parameter gradients
                    optimizer.zero_grad()

                    # forward + backward + optimize
                    outputs = self.forward(images)

                    # calculate the testing accuracy
                    accuracy += self.calculate_accuracy(predictions=outputs, gt_labels=labels)
                    count += 1

                    # if use sp calculate the group dro loss otherwise the Cross-Entropy loss
                    if self.use_subpopulation:
                        loss = self.loss_function.loss(outputs, labels, sf_to_id)
                        if (step + 1) % log_every == 0:
                            self.loss_function.reset_stats()
                    else:
                        loss = self.loss_function(outputs, labels)

                    # make a backpropagation based the loss
                    loss.backward()
                    optimizer.step()

                    # Save for subpopulation metrics
                    _, predicts = torch.max(torch.softmax(outputs, dim=1), dim=1, keepdim=False)

                    predicts = predicts.cpu().detach().numpy()
                    labels = labels.cpu().detach().numpy()

                    predicts_lst.extend(predicts)
                    labels_lst.extend(labels)
                    sensitive_feature_labels_lst.extend(sf)

                    running_loss += loss.cpu().detach().item()
                    outputs = outputs.cpu().detach().numpy()

                    torch.cuda.empty_cache()
                    # print statistics
                    if (step + 1) % 10 == 0:
                        pbar.set_description(
                            f'Bath Size Step [{step}/{len(train_data_loader)}], Loss: {loss.item():.4f}')

                # calculate the training accuracy and the prediction per class
                for i in dataset.emotions_classes:
                    value = self.emotion_map[i]
                    indexes = [index for index in range(len(labels_lst)) if value == labels_lst[index]]
                    dict_class = {'predict': [predicts_lst[i] for i in indexes],
                                  'label': [labels_lst[i] for i in indexes],
                                  'sf': [sensitive_feature_labels_lst[i] for i in indexes]}
                    epoch_dict[i] = dict_class

                training_dict[epoch] = epoch_dict
                total_loss.append(running_loss / count)
                pbar.update(1)

                accuracy /= count
                accuracy_lst.append(accuracy)

        model_filename = f'model_classes_original.pt'
        folder_dir = f'use_sp_{self.use_subpopulation}_use_bn_{self.use_batch_norm}_use_LwF{False}'
        dir_path = os.getcwd()
        path_model = os.path.join(dir_path, model_dir)
        path_model = os.path.join(path_model, folder_dir)
        if not os.path.exists(path_model):
            os.mkdir(path_model)
        path_model = os.path.join(path_model, model_filename)
        print(path_model)
        self.save_model(filename=path_model)
        torch.cuda.empty_cache()
        return accuracy_lst, total_loss, training_dict

    def save_model(self, filename):
        """
        Getting the model filename and their folder directory save their dictionary
        :param filename: str, the model directory with filename
        :return: None
        """
        print(f'Save the model for for classes : {self.emotion_classes}')
        torch.save(self.state_dict(), filename)


class Model_with_LwF(Model):
    """
    This class run the Learning without forgetting algorithm and abstarct all the functions from the Model class
    """

    def __init__(self, lr, epochs, batch_size, loss_function, num_hidden=4096, weight_decay=0.0005, epsilon=1e-16,
                 total_classes=2,
                 classes=['anger', 'contempt'], use_batch_norm=False, use_pre_trained_weights=False,
                 use_subpopulation=True, sensitive_feature='race4', use_old_class_data=False):
        """
            The initialize function where the model are initialize
            :param lr: float, the learning rate of the model value
            :param epochs: int, the total number of epochs
            :param batch_size: int, the total value of the batch size
            :param num_hidden: int, the number of the hidden layer in fully connected
            :param use_batch_norm: bool, a boolean state where said if had Batch normalization the model
            :param use_pre_trained_weights: bool, A boolean state where said if must be used pre trained weights
            :param weight_decay: float, the value of weight decay for the optimizer
            :param emotion_classes: list, a list with the name of the classes
            :param use_subpopulation: bool, a boolean statement where said if use subpopulation shift with group-DRO
            :param sensitive_feature: str, the name of the sensitive feature (works only for race4 and age )
        """
        # abstract the model class
        super().__init__(lr=lr,
                         epochs=epochs,
                         batch_size=batch_size,
                         use_batch_norm=use_batch_norm,
                         use_pre_trained_weights=use_pre_trained_weights,
                         weight_decay=weight_decay,
                         emotion_classes=classes,
                         use_subpopulation=use_subpopulation,
                         sensitive_feature=sensitive_feature,
                         loss_function=loss_function)

        self.testing_accuracy_per_classes_update = []
        self.use_old_class_data = use_old_class_data
        # Constant to provide numerical stability while normalizing
        self.epsilon = epsilon
        self.num_classes = total_classes
        self.emotion_classes = classes
        self.emotion_map = emotion_map

        self.accuracy_thresh_decay = 0.8
        self.total_hidden_layers = num_hidden

        # Creating the model
        self.model_architecture = VGG_19_model_architecture

        self.feature_extractor = self.create_feature_extractor()
        self.fc = self.create_fully_connected_layer()
        self.fc.apply(kaiming_normal_init)

        if self.use_pre_trained_weights:
            self.add_pre_trained_weights()
        else:
            self.feature_extractor.apply(kaiming_normal_init)

    def add_new_classes(self, new_classes):
        """
        Add a new class or classes to the model output layer
        :param new_classes:
        :return: None
        """
        new_class_total = len(new_classes)
        old_classes_total = self.num_classes
        self.num_classes += new_class_total
        self.emotion_classes = [*self.emotion_classes, *new_classes]

        old_class_weights = self.fc.output.weight.data
        old_class_bias = self.fc.output.bias.data
        # add the new layer and copy the old classes weights
        self.fc.output = nn.Linear(self.total_hidden_layers, self.num_classes).to(self.device)
        self.fc.output.apply(kaiming_normal_init)

        self.fc.output.weight.data[:old_classes_total, :] = old_class_weights
        self.fc.output.bias.data[:old_classes_total] = old_class_bias

    def testing(self, testing_dataset, prev_model=None, use_prev_model=False, T=2, beta=0.25):
        """
        Test model architecture and return the testing accuracy(This function can call and for validation)
        :param testing_dataset: torch.utils.data.Dataset, the Dataset class
        :param prev_model: Model/None, the previous model if evaluate or testing for a new class
        :param use_prev_model: bool, if have an old class model
        :param T: int, the T of the Distillation loss
        :param beta: float, the beta is the weight value for how much to count the distillation loss on the total loss
        :return: float, float: The testing/evaluation accuracy and loss
        """
        # Load the testing dataset
        testing_data_loader = data.DataLoader(dataset=testing_dataset,
                                              batch_size=self.batch_size,
                                              shuffle=False,
                                              num_workers=0)
        with torch.no_grad():
            count = 0
            accuracy = 0
            total_loss = 0
            for step, datas in enumerate(testing_data_loader):
                images, labels, sf = datas
                images = images.to(self.device)
                labels = labels.to(self.device)

                if self.sensitive_feature == 'race4':
                    sf_to_id = [races4_to_id[i] for i in sf]
                elif self.sensitive_feature == 'age':
                    sf_to_id = [binary_age_groups_to_id[i] for i in sf]

                # Make prediction using forward
                predicts = self.forward(images)

                # Predict the loss based the prediction and the gt label
                # If use SP calculate the group-dro loss otherwise the Cross-Entropy
                if self.use_subpopulation:
                    loss = self.loss_function.loss(predicts, labels, sf_to_id)
                else:
                    loss = self.loss_function(predicts, labels)

                # If have and an old model calculate the knowledge distillation loss
                if use_prev_model:
                    old_class_outputs = prev_model.forward(images)
                    old_class_size = old_class_outputs.shape[1]

                    loss_2 = nn.KLDivLoss()(F.log_softmax(predicts[:, :old_class_size] / T, dim=1),
                                            F.softmax(old_class_outputs.detach() / T,
                                                      dim=1)) * T * T * beta * old_class_size

                    loss += loss_2

                total_loss += loss
                accuracy += self.calculate_accuracy(predictions=predicts, gt_labels=labels)
                count += 1
                predicts = predicts.cpu().detach().numpy()
                labels = labels.cpu().detach().numpy()

                torch.cuda.empty_cache()
            return accuracy / count, total_loss / count

    def calculate_all_classes_testing_accuracy(self, datasets, test_dataset=True):
        """
        This class calculate the accuracy and for the old classes and for the new added
        :param test_dataset: bool, identify if th testing data is used for evaluation or the validation data
        :param datasets: list, a list with all the dataset
        :return: dict: a dictionary for the prediction for each class ground truth
        """
        testing_dict = {}
        for dataset in datasets:
            if test_dataset:
                testing_data_loader = data.DataLoader(dataset=dataset.testing_dataset,
                                                      batch_size=self.batch_size,
                                                      shuffle=False,
                                                      num_workers=0)
            else:
                testing_data_loader = data.DataLoader(dataset=dataset.validation_dataset,
                                                      batch_size=self.batch_size,
                                                      shuffle=False,
                                                      num_workers=0)
            predicts_lst = []
            labels_lst = []
            sensitive_feature_labels_lst = []
            with torch.no_grad():
                for datas in testing_data_loader:
                    images, labels, sf = datas
                    images = images.to(self.device)
                    labels = labels.to(self.device)
                    predicts = self.classify(images)

                    predicts = predicts.cpu().detach().numpy()
                    labels = labels.cpu().detach().numpy()

                    predicts_lst.extend(predicts)
                    labels_lst.extend(labels)
                    sensitive_feature_labels_lst.extend(sf)
                    torch.cuda.empty_cache()

            # for each class add the predictions
            for i in dataset.emotions_classes:
                value = self.emotion_map[i]
                indexes = [index for index in range(len(labels_lst)) if value == labels_lst[index]]
                dict_class = {'predict': [predicts_lst[i] for i in indexes],
                              'label': [labels_lst[i] for i in indexes],
                              'sf': [sensitive_feature_labels_lst[i] for i in indexes]}
                testing_dict[i] = dict_class

        return testing_dict

    def train_old_classes(self, dataset, log_every=50):
        """
        Train the model for the N old classes
        :param dataset: DataLoader_Affect_Net, the dataloader of the AffectNet with the training, test and validation dataset
        :param log_every: int, a number to say how every step in batch size to reset the loss function of group-DRO when is used
        :return: None
        """
        training_dict = {}
        total_loss = []
        accuracy_lst = []
        train_data_loader = data.DataLoader(dataset=dataset.training_dataset,
                                            batch_size=self.batch_size,
                                            shuffle=False,
                                            num_workers=0)

        # initialize the SGD optimizer
        optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, self.parameters()), lr=self.lr,
                                    momentum=self.momentum, weight_decay=5e-4)

        # Add fine tuning to revert from the overfitting
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=1. / 3, patience=5)
        with tqdm(total=self.epochs) as pbar:
            epoch_dict = {}
            for epoch in range(self.epochs):
                running_loss = 0.0
                accuracy = 0
                count = 0
                predicts_lst = []
                labels_lst = []
                sensitive_feature_labels_lst = []
                for step, datas in enumerate(train_data_loader):
                    images, labels, sf = datas
                    images = images.to(self.device)
                    labels = labels.to(self.device)

                    # make the sf from str to id
                    if self.sensitive_feature == 'race4':
                        sf_to_id = [races4_to_id[i] for i in sf]
                    elif self.sensitive_feature == 'age':
                        sf_to_id = [binary_age_groups_to_id[i] for i in sf]

                    # zero the parameter gradients
                    optimizer.zero_grad()

                    # forward + backward + optimize
                    outputs = self.forward(images)

                    # calculate the testing accuracy
                    accuracy += self.calculate_accuracy(predictions=outputs, gt_labels=labels)
                    count += 1

                    # if use sp calculate the group dro loss otherwise the Cross-Entropy loss
                    if self.use_subpopulation:
                        loss = self.loss_function.loss(outputs, labels, sf_to_id)
                        if (step + 1) % log_every == 0:
                            self.loss_function.reset_stats()
                    else:
                        loss = self.loss_function(outputs, labels)

                    # make a backpropagation based the loss
                    loss.backward()
                    optimizer.step()

                    # Save for subpopulation metrics
                    _, predicts = torch.max(torch.softmax(outputs, dim=1), dim=1, keepdim=False)

                    predicts = predicts.cpu().detach().numpy()
                    labels = labels.cpu().detach().numpy()

                    predicts_lst.extend(predicts)
                    labels_lst.extend(labels)
                    sensitive_feature_labels_lst.extend(sf)

                    running_loss += loss.cpu().detach().item()
                    outputs = outputs.cpu().detach().numpy()

                    torch.cuda.empty_cache()
                    # print statistics
                    if (step + 1) % 10 == 0:
                        pbar.set_description(
                            f'Bath Size Step [{step}/{len(train_data_loader)}], Loss: {loss.item():.4f}')

                # calculate the validation acc and loss
                self.eval()
                val_acc, val_loss = self.testing(dataset.validation_dataset)
                self.train()
                sched.step(val_loss)

                # calculate the training accuracy and the prediction per class
                for i in dataset.emotions_classes:
                    value = self.emotion_map[i]
                    indexes = [index for index in range(len(labels_lst)) if value == labels_lst[index]]
                    dict_class = {'predict': [predicts_lst[i] for i in indexes],
                                  'label': [labels_lst[i] for i in indexes],
                                  'sf': [sensitive_feature_labels_lst[i] for i in indexes]}
                    epoch_dict[i] = dict_class

                training_dict[epoch] = epoch_dict
                total_loss.append(running_loss / count)
                pbar.update(1)

                accuracy /= count
                accuracy_lst.append(accuracy)

        model_filename = f'model_classes_{len(self.emotion_classes)}.pt'
        folder_dir = f'use_sp_{self.use_subpopulation}_use_bn_{self.use_batch_norm}'
        dir_path = os.getcwd()
        path_model = os.path.join(dir_path, model_dir)
        path_model = os.path.join(path_model, folder_dir)
        if not os.path.exists(path_model):
            os.mkdir(path_model)
        path_model = os.path.join(path_model, model_filename)
        print(path_model)
        self.save_model(filename=path_model)
        torch.cuda.empty_cache()
        return accuracy_lst, total_loss, training_dict

    def training_(self, dataset, preprocessing, new_classes=[], new_classes_itter=2, T=2, log_every=50):
        """
        Training the first N classes and apply LwF for each pair of the new classes
        :param dataset: Dataset, the dataset for the N first old classes
        :param preprocessing: Preprocessing, a class with a preprocessing for the subpopulation when is used
        :param new_classes: list, a list with the new classes
        :param new_classes_itter: int, the number for the adition of the new classes each time
        :param T: int, the value for the Knowledge distillation loss
        :param log_every: int, how many time to clean the attributes of the group-dro
        :return: list, list, dict: The predictions for the plots
        """
        # If new classes is empty train only on the old classes
        beta = 0.75
        accuracy_all_classes = []
        loss_all_classes = []
        testing_accuracy_per_class = []
        testing_datasets = []
        training_accuracy_per_class = []
        if len(new_classes) == 0:
            # if select to train all the classes and don't have Lwf
            accuracy_old_classes, loss_old_classes, training_dict = self.train_old_classes(dataset=dataset)
            training_accuracy_per_class.append(training_dict)
            accuracy_all_classes.append(accuracy_old_classes)
            loss_all_classes.append(loss_old_classes)
            # Test accuracy and loss for the old classes
            test_acc, test_loss = self.testing(dataset.testing_dataset)
            testing_accuracy_per_class.append(test_acc)

            testing_datasets.append(dataset)
            self.testing_accuracy_per_classes_update.append(
                self.calculate_all_classes_testing_accuracy(datasets=testing_datasets, test_dataset=True))
            return accuracy_all_classes, loss_all_classes, testing_accuracy_per_class, training_accuracy_per_class
        else:
            print(f'Train on classes {self.emotion_classes}\n')
            accuracy_old_classes, loss_old_classes, training_dict = self.train_old_classes(dataset=dataset)
            training_accuracy_per_class.append(training_dict)
            accuracy_all_classes.append(accuracy_old_classes)
            loss_all_classes.append(loss_old_classes)
            # Test accuracy and loss for the old classes
            test_acc, test_loss = self.testing(dataset.testing_dataset)
            testing_accuracy_per_class.append(test_acc)

            testing_datasets.append(dataset)
            self.testing_accuracy_per_classes_update.append(
                self.calculate_all_classes_testing_accuracy(datasets=testing_datasets, test_dataset=True))
            for new_emotion_classes in zip(*[iter(new_classes)] * new_classes_itter):
                training_dict = {}
                torch.cuda.empty_cache()
                # load the old class model
                folder_dir = f'use_sp_{self.use_subpopulation}_use_bn_{self.use_batch_norm}'
                model_filename = f'model_classes_{len(self.emotion_classes)}.pt'
                dir_path = os.getcwd()

                # Load old class model
                path_model = os.path.join(dir_path, model_dir)
                path_model = os.path.join(path_model, folder_dir)
                path_model = os.path.join(path_model, model_filename)
                prev_model = Model(lr=self.lr,
                                   epochs=self.epochs,
                                   batch_size=self.batch_size,
                                   use_batch_norm=self.use_batch_norm,
                                   use_pre_trained_weights=self.use_pre_trained_weights,
                                   weight_decay=self.weight_decay,
                                   emotion_classes=self.emotion_classes,
                                   loss_function=self.loss_function)

                checkpoints_old_model = torch.load(path_model, map_location=self.device)
                prev_model.load_state_dict(checkpoints_old_model)

                # freeze old class model
                prev_model.eval()
                for param in prev_model.parameters():
                    param.requires_grad = False
                new_emotion_classes = list(new_emotion_classes)
                print(f'Add class {new_emotion_classes}\n')
                del checkpoints_old_model

                # add the new class and change the fully connected layer
                self.add_new_classes(new_classes=new_emotion_classes)

                # Add SGD optimizer for the share parameters
                optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, self.parameters()),
                                            lr=self.lr / 2,
                                            momentum=self.momentum, weight_decay=5e-4)
                # add fine tuning to don't affect from overfitting
                sched = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=1. / 3, patience=5)
                dataset_new_class = DataLoader_Affect_Net(classes=new_emotion_classes,
                                                          use_subpopulation=self.use_subpopulation,
                                                          pre_processing=preprocessing,
                                                          sensitive_feature=dataset.sensitive_feature)
                if self.use_old_class_data:
                    old_classes = [emotion for emotion in self.emotion_classes if emotion not in new_classes]
                    dataset_new_class.merge_training_data(emotion_classes=old_classes, datasets=testing_datasets)

                train_data_loader = data.DataLoader(dataset=dataset_new_class.training_dataset,
                                                    batch_size=self.batch_size,
                                                    shuffle=False,
                                                    num_workers=0)

                total_loss = []
                accuracy_lst = []

                testing_datasets.append(dataset_new_class)
                if self.use_subpopulation:
                    self.loss_function.update_dataset(dataset=dataset_new_class.training_dataset)

                self.train()
                with tqdm(total=self.epochs) as pbar:
                    epoch_dict = {}
                    for epoch in range(self.epochs):
                        running_loss = 0.0
                        accuracy = 0
                        count = 0
                        predicts_lst = []
                        labels_lst = []
                        sensitive_feature_labels_lst = []
                        torch.cuda.empty_cache()

                        for step, datas in enumerate(train_data_loader):
                            images, labels, sf = datas
                            images = images.to(self.device)
                            labels = labels.to(self.device)
                            # zero the parameter gradients
                            optimizer.zero_grad()
                            # convert sf from string to id
                            if self.sensitive_feature == 'race4':
                                sf_to_id = [races4_to_id[i] for i in sf]
                            elif self.sensitive_feature == 'age':
                                sf_to_id = [binary_age_groups_to_id[i] for i in sf]
                            # get the prediction and calculate the loss
                            outputs = self.forward(images)
                            if self.use_subpopulation:
                                loss_1 = self.loss_function.loss(outputs, labels, sf_to_id)
                                if (step + 1) % log_every == 0:
                                    self.loss_function.reset_stats()
                            else:
                                loss_1 = self.loss_function(outputs, labels)

                            old_class_outputs = prev_model.forward(images)
                            old_class_size = old_class_outputs.shape[1]
                            # add Knowledge distillation loss based old class parameters
                            loss_2 = nn.KLDivLoss()(F.log_softmax(outputs[:, :old_class_size] / T, dim=1),
                                                    F.softmax(old_class_outputs.detach() / T,
                                                              dim=1)) * T * T * beta * old_class_size
                            # add two loss
                            loss = loss_1 + loss_2
                            accuracy += self.calculate_accuracy(predictions=outputs, gt_labels=labels)
                            count += 1
                            # Back propagation for Knowledge distillation loss

                            loss.backward()
                            optimizer.step()

                            # print statistics
                            running_loss += loss.cpu().detach().item()

                            # Save for subpopulation metrics
                            _, predicts = torch.max(torch.softmax(outputs, dim=1), dim=1, keepdim=False)

                            predicts = predicts.cpu().detach().numpy()
                            labels = labels.cpu().detach().numpy()

                            predicts_lst.extend(predicts)
                            labels_lst.extend(labels)
                            sensitive_feature_labels_lst.extend(sf)

                            if step % 10 == 0:
                                pbar.set_description(
                                    f'Bath Size Step [{step}/{len(train_data_loader)}, Loss: {loss.item():.4f}')
                            torch.cuda.empty_cache()

                        self.eval()
                        val_acc, val_loss = self.testing(dataset.validation_dataset,
                                                         prev_model=prev_model, use_prev_model=True,
                                                         T=T, beta=beta)
                        self.train()
                        sched.step(val_loss)

                        pbar.update(1)
                        total_loss.append(running_loss / count)
                        accuracy /= count

                        accuracy_lst.append(accuracy)
                        for i in dataset.emotions_classes:
                            value = self.emotion_map[i]
                            indexes = [index for index in range(len(labels_lst)) if value == labels_lst[index]]
                            dict_class = {'predict': [predicts_lst[i] for i in indexes],
                                          'label': [labels_lst[i] for i in indexes],
                                          'sf': [sensitive_feature_labels_lst[i] for i in indexes]}
                            epoch_dict[i] = dict_class

                        training_dict[epoch] = epoch_dict
                        self.train()

                    training_accuracy_per_class.append(training_dict)
                    # save the new class model
                    model_filename = f'model_classes_{len(self.emotion_classes)}.pt'
                    folder_dir = f'use_sp_{self.use_subpopulation}_use_bn_{self.use_batch_norm}'
                    dir_path = os.getcwd()
                    path_model = os.path.join(dir_path, model_dir)
                    path_model = os.path.join(path_model, folder_dir)

                    if not os.path.exists(path_model):
                        os.mkdir(path_model)

                    path_model = os.path.join(path_model, model_filename)
                    print(path_model)
                    self.save_model(filename=path_model)
                    self.eval()
                    self.testing_accuracy_per_classes_update.append(
                        self.calculate_all_classes_testing_accuracy(datasets=testing_datasets, test_dataset=True))

                    # calculate the metrics for the plots
                    accuracy_all_classes.append(accuracy_lst)
                    loss_all_classes.append(total_loss)
                    test_acc, test_loss = self.testing(dataset_new_class.testing_dataset)
                    testing_accuracy_per_class.append(test_acc)
            return accuracy_all_classes, loss_all_classes, testing_accuracy_per_class, training_accuracy_per_class
