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


def kaiming_normal_init(model):
    if isinstance(model, nn.Conv2d):
        nn.init.kaiming_normal_(model.weight, nonlinearity='relu')
    elif isinstance(model, nn.Linear):
        nn.init.kaiming_normal_(model.weight, nonlinearity='sigmoid')


def knowledge_distillation_loss(outputs, labels, T=2):
    # T equal with 2 like LwF paper
    outputs_sf = torch.log_softmax(outputs / T, dim=1)
    labels_sf = torch.softmax(labels / T, dim=1)

    outputs_sum = torch.sum(outputs_sf * labels_sf, dim=1, keepdim=False)
    outputs_mean = -torch.mean(outputs_sum, dim=0, keepdim=False)
    outputs_mean = outputs_mean * T * T
    return Variable(outputs_mean.data, requires_grad=True)


class Model(nn.Module):

    def __init__(self, lr, epochs, batch_size, num_hidden=4096, use_batch_norm=False, use_pre_trained_weights=False,
                 weight_decay=0.0005,
                 emotion_classes=['anger', 'contempt', 'disgust', 'fear', 'sad', 'surprise', 'happy', 'neutral'],
                 use_subpopulation=True):
        # Initialize Hyper parameters
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size

        self.use_batch_norm = use_batch_norm
        self.use_pre_trained_weights = use_pre_trained_weights
        # Learning without forgetting parameters
        self.momentum = 0.9
        self.weight_decay = weight_decay

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

        if self.use_pre_trained_weights:
            self.add_pre_trained_weights()
        else:
            self.feature_extractor.apply(kaiming_normal_init)

    def classify(self, images):
        _, preds = torch.max(torch.softmax(self.forward(images), dim=1), dim=1, keepdim=False)

        return preds

    def add_pre_trained_weights(self):
        if self.use_batch_norm:
            vgg_19 = torchvision.models.vgg19_bn(pretrained=True, progress=False)
        else:
            vgg_19 = torchvision.models.vgg19(pretrained=True, progress=False)
        for param1, param2 in zip(vgg_19.features, self.feature_extractor):
            if isinstance(param1, nn.Conv2d) or isinstance(param1, nn.BatchNorm2d):
                param2.weight.data = param1.weight.data
                param2.bias.data = param1.bias.data

        self.feature_extractor = self.feature_extractor.to(self.device)

    def calculate_accuracy(self, predictions, gt_labels):
        _, predicts = torch.max(torch.softmax(predictions, dim=1), dim=1, keepdim=False)
        corrects = (predicts == gt_labels).sum()
        corrects = corrects.cpu().detach().numpy()

        return corrects / gt_labels.size(0)

    def forward(self, img_x):
        x = self.feature_extractor(img_x)
        x = x.view(x.size(0), -1)
        out = self.fc(x)
        return out

    def create_feature_extractor(self):
        layers = []
        input_channels = self.colour_channel

        for layer in self.model_architecture:
            if type(layer) == int:
                out_channel = layer
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

        layers_Features_extraction = nn.Sequential(*layers).to(self.device)
        return layers_Features_extraction

    def create_fully_connected_layer(self):
        total_max_pool = self.model_architecture.count("M")
        factor = 2 ** total_max_pool

        if self.input_width % factor != 0 and self.input_height % factor != 0:
            raise ValueError(f"`in_height` and `in_width` must be multiples of {factor}")
        output_height = self.input_height // factor
        output_width = self.input_width // factor
        last_output_channel = next(
            layer_size for layer_size in self.model_architecture[::-1] if type(layer_size) == int)
        self.fc_input_size = last_output_channel * output_height * output_width

        fc_layer = nn.Sequential(OrderedDict([
            ('fc1', nn.Linear(self.fc_input_size, self.total_hidden_layers)),
            ('relu1', nn.ReLU()),
            ('dropout1', nn.Dropout(p=0.5)),
            ('fc2', nn.Linear(self.total_hidden_layers, self.total_hidden_layers)),
            ('relu2', nn.ReLU()),
            ('dropout2', nn.Dropout(p=0.5)),
            ('output', nn.Linear(self.total_hidden_layers, self.num_classes))
        ])).to(self.device)
        return fc_layer

    def training_(self, dataset):
        total_loss = []
        train_data_loader = data.DataLoader(dataset=dataset.training_dataset, batch_size=self.batch_size, shuffle=False,
                                            num_workers=0)
        criterion = nn.CrossEntropyLoss()
        # Observe that all parameters are being optimized

        optimizer = torch.optim.SGD(self.parameters(), lr=self.lr, momentum=self.momentum,
                                    weight_decay=self.weight_decay)
        with tqdm(total=self.epochs) as pbar:
            for epoch in range(self.epochs):
                running_loss = 0.0
                for step, datas in enumerate(train_data_loader):
                    images, labels, sf = datas
                    images = images.to(self.device)
                    labels = labels.to(self.device)

                    # zero the parameter gradients
                    optimizer.zero_grad()

                    # forward + backward + optimize
                    outputs = self.forward(images)

                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()

                    # print statistics
                    running_loss += loss.item()
                    if (step + 1) % 10 == 0:
                        pbar.set_description(
                            f'Bath Size Step [{step}/{len(train_data_loader)}], Loss: {loss.item():.4f}')
                total_loss.append(total_loss)
                pbar.update(1)

    def save_model(self, filename):
        print(f'Save the model for for classes : {self.emotion_classes}')
        torch.save(self.state_dict(), filename)


class Model_with_LwF(Model):

    def __init__(self, lr, epochs, batch_size, num_hidden=4096, weight_decay=0.0005, epsilon=1e-16, total_classes=2,
                 classes=['anger', 'contempt'], use_batch_norm=False, use_pre_trained_weights=False,
                 use_subpopulation=True):
        super().__init__(lr=lr,
                         epochs=epochs,
                         batch_size=batch_size,
                         use_batch_norm=use_batch_norm,
                         use_pre_trained_weights=use_pre_trained_weights,
                         weight_decay=weight_decay,
                         emotion_classes=classes,
                         use_subpopulation=use_subpopulation)

        self.testing_accuracy_per_classes_update = []

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
        new_class_total = len(new_classes)
        old_classes_total = self.num_classes
        self.num_classes += new_class_total
        self.emotion_classes = [*self.emotion_classes, *new_classes]

        old_class_weights = self.fc.output.weight.data
        old_class_bias = self.fc.output.bias.data

        self.fc.output = nn.Linear(self.total_hidden_layers, self.num_classes).to(self.device)
        self.fc.apply(kaiming_normal_init)
        self.fc.output.weight.data[:old_classes_total, :] = old_class_weights
        self.fc.output.bias.data[:old_classes_total] = old_class_bias

        # for param in self.feature_extractor.parameters():
        #     param.requires_grad = False

    def testing(self, testing_dataset):
        testing_data_loader = data.DataLoader(dataset=testing_dataset,
                                              batch_size=self.batch_size,
                                              shuffle=False,
                                              num_workers=0)
        with torch.no_grad():
            count = 0
            accuracy = 0
            for step, datas in enumerate(testing_data_loader):
                images, labels, sf = datas
                images = images.to(self.device)
                labels = labels.to(self.device)

                predicts = self.forward(images)
                accuracy += self.calculate_accuracy(predictions=predicts, gt_labels=labels)
                count += 1

            return accuracy / count

    def calculate_all_classes_testing_accuracy(self, datasets):
        testing_dict = {}
        for dataset in datasets:
            testing_data_loader = data.DataLoader(dataset=dataset.testing_dataset,
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

            for i in dataset.emotions_classes:
                value = self.emotion_map[i]
                indexes = [index for index in range(len(labels_lst)) if value == labels_lst[index]]
                dict_class = {'predict': [predicts_lst[i] for i in indexes],
                              'label': [labels_lst[i] for i in indexes],
                              'sf': [sensitive_feature_labels_lst[i] for i in indexes]}
                testing_dict[i] = dict_class

        return testing_dict

    def train_old_classes(self, dataset):
        training_dict = {}
        total_loss = []
        accuracy_lst = []
        train_data_loader = data.DataLoader(dataset=dataset.training_dataset,
                                            batch_size=self.batch_size,
                                            shuffle=False,
                                            num_workers=0)

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, self.parameters()), lr=self.lr,
                                    momentum=self.momentum, weight_decay=5e-4)

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

                    # zero the parameter gradients
                    optimizer.zero_grad()

                    # forward + backward + optimize
                    outputs = self.forward(images)

                    accuracy += self.calculate_accuracy(predictions=outputs, gt_labels=labels)
                    count += 1

                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()

                    # Save for subpopulation metrics
                    _, predicts = torch.max(torch.softmax(outputs, dim=1), dim=1, keepdim=False)

                    predicts = predicts.cpu().detach().numpy()
                    labels = labels.cpu().detach().numpy()

                    predicts_lst.extend(predicts)
                    labels_lst.extend(labels)
                    sensitive_feature_labels_lst.extend(sf)

                    # print statistics
                    running_loss += loss.item()
                    if (step + 1) % 10 == 0:
                        pbar.set_description(
                            f'Bath Size Step [{step}/{len(train_data_loader)}], Loss: {loss.item():.4f}')

                for i in dataset.emotions_classes:
                    value = self.emotion_map[i]
                    indexes = [index for index in range(len(labels_lst)) if value == labels_lst[index]]
                    dict_class = {'predict': [predicts_lst[i] for i in indexes],
                                  'label': [labels_lst[i] for i in indexes],
                                  'sf': [sensitive_feature_labels_lst[i] for i in indexes]}
                    epoch_dict[i] = dict_class

                training_dict[epoch] = epoch_dict
                total_loss.append(running_loss)
                pbar.update(1)

                accuracy /= count
                if accuracy > self.accuracy_thresh_decay:
                    for g in optimizer.param_groups:
                        g['lr'] /= 10

                accuracy_lst.append(accuracy)

        model_filename = f'model_classes_{len(self.emotion_classes)}.pt'
        dir_path = os.getcwd()
        path_model = os.path.join(dir_path, model_dir)
        path_model = os.path.join(path_model, model_filename)
        print(path_model)
        self.save_model(filename=path_model)
        return accuracy_lst, total_loss, training_dict

    def training_(self, dataset, preprocessing, new_classes=[], new_classes_itter=2, T=2):
        # If new classes is empty train only on the old classes
        beta = 0.25
        accuracy_all_classes = []
        loss_all_classes = []
        testing_accuracy_per_class = []
        testing_datasets = []
        training_accuracy_per_class = []
        if len(new_classes) == 0:
            self.train_old_classes(dataset=dataset)
            testing_accuracy_per_class.append(self.testing(dataset.testing_dataset))
        else:
            print(f'Train on classes {self.emotion_classes}\n')
            accuracy_old_classes, loss_old_classes, training_dict = self.train_old_classes(dataset=dataset)
            training_accuracy_per_class.append(training_dict)
            accuracy_all_classes.append(accuracy_old_classes)
            loss_all_classes.append(loss_old_classes)
            testing_accuracy_per_class.append(self.testing(dataset.testing_dataset))

            testing_datasets.append(dataset)
            self.testing_accuracy_per_classes_update.append(
                self.calculate_all_classes_testing_accuracy(datasets=testing_datasets))
            for new_classes in zip(*[iter(new_classes)] * new_classes_itter):
                training_dict = {}
                torch.cuda.empty_cache()
                model_filename = f'model_classes_{len(self.emotion_classes)}.pt'
                dir_path = os.getcwd()
                path_model = os.path.join(dir_path, model_dir)
                path_model = os.path.join(path_model, model_filename)
                prev_model = Model(lr=self.lr,
                                   epochs=self.epochs,
                                   batch_size=self.batch_size,
                                   use_batch_norm=self.use_batch_norm,
                                   use_pre_trained_weights=self.use_pre_trained_weights,
                                   weight_decay=self.weight_decay,
                                   emotion_classes=self.emotion_classes)

                checkpoints_old_model = torch.load(path_model, map_location=self.device)
                prev_model.load_state_dict(checkpoints_old_model)
                # prev_model = prev_model.to(self.device)
                prev_model.eval()
                for param in prev_model.parameters():
                    param.requires_grad = False
                new_classes = list(new_classes)
                print(f'Add class {new_classes}\n')
                # add the new class and change the fully connected layer

                self.add_new_classes(new_classes=new_classes)
                dataset_new_class = DataLoader_Affect_Net(classes=new_classes,
                                                          use_subpopulation=self.use_subpopulation,
                                                          pre_processing=preprocessing,
                                                          sensitive_feature=dataset.sensitive_feature)

                # old_classes = [emotion for emotion in self.emotion_classes if emotion not in new_classes]
                # dataset_new_class.merge_training_data(emotion_classes=old_classes, datasets=testing_datasets)

                train_data_loader = data.DataLoader(dataset=dataset_new_class.training_dataset,
                                                    batch_size=self.batch_size,
                                                    shuffle=False,
                                                    num_workers=0)

                total_loss = []
                accuracy_lst = []
                # Every new classes decay the learning rate
                self.lr = self.lr/2
                optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, self.parameters()), lr=self.lr,
                                            momentum=self.momentum, weight_decay=5e-4)

                criterion = nn.CrossEntropyLoss()
                testing_datasets.append(dataset_new_class)

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
                        for step, datas in enumerate(train_data_loader):
                            images, labels, sf = datas
                            images = images.to(self.device)
                            labels = labels.to(self.device)
                            # zero the parameter gradients
                            optimizer.zero_grad()

                            outputs = self.forward(images)
                            loss_1 = criterion(outputs, labels)

                            old_class_outputs = prev_model.forward(images)
                            old_class_size = old_class_outputs.shape[1]

                            # outputs_distillation = outputs[..., :old_class_size]
                            # k_d_loss = knowledge_distillation_loss(outputs_distillation, old_class_outputs)
                            # k_d_loss = k_d_loss.to(self.device)
                            loss_2 = nn.KLDivLoss()(F.log_softmax(outputs[:, :old_class_size] / T, dim=1),
                                                    F.softmax(old_class_outputs.detach() / T,
                                                              dim=1)) * T * T * beta * old_class_size
                            loss = loss_1 + loss_2

                            accuracy += self.calculate_accuracy(predictions=outputs, gt_labels=labels)
                            count += 1

                            loss.backward(retain_graph=True)
                            optimizer.step()

                            # print statistics
                            running_loss += loss.item()

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

                        pbar.update(1)
                        total_loss.append(running_loss)
                        accuracy /= count
                        if accuracy > self.accuracy_thresh_decay:
                            for g in optimizer.param_groups:
                                g['lr'] /= 10

                        accuracy_lst.append(accuracy)
                        for i in dataset.emotions_classes:
                            value = self.emotion_map[i]
                            indexes = [index for index in range(len(labels_lst)) if value == labels_lst[index]]
                            dict_class = {'predict': [predicts_lst[i] for i in indexes],
                                          'label': [labels_lst[i] for i in indexes],
                                          'sf': [sensitive_feature_labels_lst[i] for i in indexes]}
                            epoch_dict[i] = dict_class

                        training_dict[epoch] = epoch_dict

                    training_accuracy_per_class.append(training_dict)
                    model_filename = f'model_classes_{len(self.emotion_classes)}.pt'
                    dir_path = os.getcwd()
                    path_model = os.path.join(dir_path, model_dir)
                    path_model = os.path.join(path_model, model_filename)
                    print(path_model)
                    self.save_model(filename=path_model)
                    self.eval()
                    self.testing_accuracy_per_classes_update.append(
                        self.calculate_all_classes_testing_accuracy(datasets=testing_datasets))

                    accuracy_all_classes.append(accuracy_lst)
                    loss_all_classes.append(total_loss)
                    testing_accuracy_per_class.append(self.testing(dataset_new_class.testing_dataset))
            return accuracy_all_classes, loss_all_classes, testing_accuracy_per_class, training_accuracy_per_class
