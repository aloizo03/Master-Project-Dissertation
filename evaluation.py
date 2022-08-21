"""
This file script is for the model evaluation using the validation data
"""
from Dataset import DataLoader_Affect_Net
from Model import Model, Model_with_LwF
from dataset_preprocessing import pre_processing
from Plots import *
from config import *
import argparse
from loss import LossComputer
from torch.utils import data
import torch
from torch import nn
import os


def plot_all(model, acc_list, image_with_labels, sf='race4'):
    """
    Based the evaluation result plot all the results
    :param model: nn.Module, the model of learning with forgetting with all validation predictions information
    :param acc_list: list, a list with all new classes mean accuracy
    :param image_with_labels: list, a list with dictionary with the image data and the labels informations
    :param sf: str, the sensitive feature group can be race4 or age
    :return: None
    """
    print('\n\nStart Plotting')
    plot_testing_accuracy_per_classes(acc_list, classes_iter=2, classes=emotions_classes)
    plot_new_classes_testing_accuracy_per_sf(model, classes_iter=2, classes=emotions_classes, sf_group=sf)
    plot_image_and_labels(image_with_labels_lst=image_with_labels, sf=sf)


def get_first_i_data(dataset, model, batch_size, sf='race4', i=5):
    """
    Return the peridiciton for the first i data for each sensitive feature group for each class pair
    :param dataset: Dataset, the class dataset of validation dataset data
    :param model: nn.Module, the model class
    :param batch_size: int, the number of the batch size
    :param sf: str, the sensitive feature group can be race4 or age
    :param i: int, the number of images
    :return: list, a list with dictionaries for each class pair with all the images data and for each image labels and predictions
    """
    if sf == 'race4':
        sf_domains = ['Asian', 'Black', 'Indian', 'White']
    else:
        sf_domains = ['young', 'old']

    testing_data_loader = data.DataLoader(dataset=dataset,
                                          batch_size=batch_size,
                                          shuffle=False,
                                          num_workers=0)
    emotion_map = {0: 'anger', 1: 'sad',  2: 'disgust',  3: 'fear',  4: 'contempt', 5: 'surprise', 6: 'happy',
                   7: 'neutral'}
    with torch.no_grad():
        pred_dict = {}
        for domain in sf_domains:
            domain_dict = {}
            img_dict_list = []
            pred_dict_lst = []
            label_dict_lst = []
            sf_dict_lst = []
            img_count = i
            for step, datas in enumerate(testing_data_loader):
                images, labels, sf = datas
                images = images.to(model.device)
                labels = labels.to(model.device)
                predictions = model.classify(images)
                predictions_lst = predictions.tolist()
                indexes = [c for c, i in enumerate(sf) if i == domain]

                labels = labels.detach().cpu().numpy()

                if len(indexes) >= img_count:
                    indexes = indexes[:img_count]
                    indexes_2 = [batch_size * step + index for index in indexes]
                    img_dict_list.extend([dataset.image_path_x[index] for index in indexes_2])
                    label_dict_lst.extend([emotion_map[labels[index]] for index in indexes])
                    sf_dict_lst.extend([sf[index] for index in indexes])
                    pred_dict_lst.extend([emotion_map[predictions_lst[index]] for index in indexes])
                    break
                else:
                    indexes_2 = [batch_size * step + index for index in indexes]
                    img_dict_list.extend([dataset.image_path_x[index] for index in indexes_2])
                    label_dict_lst.extend([emotion_map[labels[index]] for index in indexes])
                    sf_dict_lst.extend([sf[index] for index in indexes])
                    pred_dict_lst.extend([emotion_map[predictions_lst[index]] for index in indexes])
                    img_count = img_count - len(indexes)

            domain_dict['images'] = img_dict_list
            domain_dict['predictions'] = pred_dict_lst
            domain_dict['labels'] = label_dict_lst
            domain_dict['sf'] = sf_dict_lst
            pred_dict[domain] = domain_dict

    return pred_dict


def main(args):
    """
    The main with the evaluation of the system
    :param args: args, the arguments setting for the evaluation experiments
    :return: None
    """
    new_classes_itter = args.num_classes
    use_bn = args.use_batch_norm
    use_subpopulation = args.use_sp
    sf = args.sf
    batch_size = args.batch_size
    num_emotion = new_classes_itter

    folder_dir = f'use_sp_{use_subpopulation}_use_bn_{use_bn}'
    dir_path = os.getcwd()
    path_model = os.path.join(dir_path, model_dir)
    path_model = os.path.join(path_model, folder_dir)
    if not os.path.exists(path_model):
        assert "The models with this configurations has not exist\n"

    print(f'The models are located at {path_model}\n\n')
    dataset_pre_process = pre_processing(sensitive_features=sf)


    datasets = []
    test_acc_lst = []
    img_predictions_plot = []
    tmp_lst = []
    for classes in zip(*[iter(emotions_classes)] * new_classes_itter):
        print(f'Evaluation on classes : {classes}\n')
        model_filename = f'model_classes_{num_emotion}.pt'
        filename_path = os.path.join(path_model, model_filename)
        classes_lst = list(classes)
        all_classes = emotions_classes[:num_emotion]
        data_loader_AffectNet = DataLoader_Affect_Net(classes=classes_lst,
                                                      pre_processing=dataset_pre_process,
                                                      use_subpopulation=use_subpopulation,
                                                      sensitive_feature=sf)
        model = Model_with_LwF(classes=all_classes,
                               lr=0.0001,
                               epochs=0,
                               total_classes=num_emotion,
                               use_subpopulation=False,
                               batch_size=batch_size,
                               use_batch_norm=use_bn,
                               sensitive_feature=sf,
                               loss_function=nn.CrossEntropyLoss())
        model.testing_accuracy_per_classes_update = tmp_lst
        datasets.append(data_loader_AffectNet)
        checkpoints_old_model = torch.load(filename_path, map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        model.load_state_dict(checkpoints_old_model)
        model.eval()
        model.testing_accuracy_per_classes_update.append(
            model.calculate_all_classes_testing_accuracy(datasets=datasets, test_dataset=False))
        test_acc, test_loss = model.testing(data_loader_AffectNet.validation_dataset)
        test_acc_lst.append(test_acc)
        img_predictions_plot.append(get_first_i_data(dataset=data_loader_AffectNet.validation_dataset,
                                                     model=model,
                                                     batch_size=batch_size,
                                                     sf=sf))
        num_emotion += new_classes_itter
        tmp_lst = model.testing_accuracy_per_classes_update

    plot_all(model, test_acc_lst, img_predictions_plot, sf)


def str_to_bool(v):
    """
    Convert an input string to boolean
    :param v: str, the input
    :return: bool, the bool value
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('A boolean value expected on the ')


def config_parse(args):
    """
    Check the command line arguments if are give correctly
    :param args: args, the argument from command line
    :return: None
    """
    if args.num_classes < 0:
        raise AssertionError('The --num_classes value cannot be a negative number')
    if args.batch_size < 0:
        raise AssertionError('The --batch_size value cannot be a negative number')
    if args.sf != 'race4' and args.sf != 'age':
        raise AssertionError('The --sf value must be race4 or age')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Learning without Forgetting for facial emotion recognition with subpopulation shift option evaluation')
    parser.add_argument('--num_classes', default=2, help='Number of new classes introduced each time to the model',
                        type=int)
    parser.add_argument('--batch_size', default=64, type=int, help='The batch size')
    parser.add_argument('--use_sp', default=True, type=str_to_bool,
                        help='Identify if we use subpopulation in our model training and testing')
    parser.add_argument('--use_batch_norm', default=False, type=str_to_bool,
                        help='Identify if we have batch normalization into '
                             'the Network model')
    parser.add_argument('--sf', default='race4', type=str,
                        help='Identify which class will be the sensitive feature (The 4 races or binary age group)\n'
                             'The value must be race4 or age')
    args = parser.parse_args()
    config_parse(args)
    main(args)
