from Dataset import DataLoader_Affect_Net
from Model import Model, Model_with_LwF
import torch.nn as nn
from dataset_preprocessing import *
from Plots import *
from config import *
import argparse
from ConfusionMatrix import ConfusionMatrix
from loss import LossComputer
import torch


def calculate_confunsion_matrix_per_sf(model,
                                       classes=['anger', 'contempt', 'disgust', 'fear', 'sad', 'surprise', 'happy',
                                                'neutral'],
                                       classes_iter=2, sf='race4'):
    """
    This taking the model create for each class pair a confusion matrix for each sensitive feature group
    :param model: nn.Module, the model class
    :param classes: list, a list with all the classes
    :param classes_iter:
    :param sf: str, which are the sensitive feature group
    :return:
    """
    cm_list = []
    for accuracy_per_new_task, classes_per_tt in zip(model.testing_accuracy_per_classes_update,
                                                     zip(*[iter(classes)] * classes_iter)):
        classes_per_tt = list(classes_per_tt)
        labels_lst = []
        preds_lst = []
        sf_lst = []

        if sf == 'race4':
            sensitive_features = list(races4)
        elif sf == 'age':
            sensitive_features = list(binary_age_groups)

        for key, value in accuracy_per_new_task.items():
            preds_lst.extend(value['predict'])
            labels_lst.extend(value['label'])
            sf_lst.extend(value['sf'])

        cm = {}
        for sf in sensitive_features:
            indexes = [index for index in range(len(sf_lst)) if sf == sf_lst[index]]
            predicts = [preds_lst[i] for i in indexes]
            labels = [labels_lst[i] for i in indexes]
            cm[sf] = ConfusionMatrix(predicts=predicts,
                                     ground_truth_labels=labels,
                                     classes=classes_per_tt)
        cm_list.append(cm)

    return cm_list


def main(args):
    """
    The main class of the executing the experiments
    :param args: args, The variables where are given from the command line for the experiment configurations
    :return: None
    """
    new_classes_itter = args.num_classes
    batch_size = args.batch_size
    lr = args.lr
    num_of_epochs = args.epochs
    use_pt_weight = args.use_pt_weights
    use_bn = args.use_batch_norm
    use_subpopulation = args.use_sp
    sf = args.sf
    use_old_class_data = args.old_class_data

    classes = emotions_classes
    if new_classes_itter > 0:
        old_classes = [classes[i] for i in range(new_classes_itter)]
        new_classes = [classes[i] for i in range(new_classes_itter, len(emotions_classes))]

    dataset_pre_process = pre_processing(sensitive_features=sf)
    data_loader_AffectNet = DataLoader_Affect_Net(classes=old_classes,
                                                  pre_processing=dataset_pre_process,
                                                  use_subpopulation=use_subpopulation,
                                                  sensitive_feature=sf)

    adjustments = [float(c) for c in generalization_adjustment]
    if len(adjustments) == 1:
        adjustments = np.array(adjustments * data_loader_AffectNet.training_dataset.n_groups)
    else:
        adjustments = np.array(adjustments)

    if use_subpopulation:
        criterion = nn.CrossEntropyLoss(reduction='none')
        train_loss_computer = LossComputer(
            criterion,
            is_robust=is_robust,
            dataset=data_loader_AffectNet.training_dataset,
            alpha=0.1,
            gamma=0.1,
            adj=adjustments,
            step_size=robust_step_size,
            normalize_loss=use_normalized_loss,
            btl=btl,
            min_var_weight=minimum_variational_weight,
            device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    else:
        criterion = nn.CrossEntropyLoss()

    if use_subpopulation:
        model = Model_with_LwF(classes=old_classes,
                               total_classes=new_classes_itter,
                               lr=lr,
                               batch_size=batch_size,
                               epochs=num_of_epochs,
                               use_pre_trained_weights=use_pt_weight,
                               use_batch_norm=use_bn,
                               use_subpopulation=use_subpopulation,
                               sensitive_feature=sf,
                               loss_function=train_loss_computer,
                               use_old_class_data=use_old_class_data)
    else:
        model = Model_with_LwF(classes=old_classes,
                               total_classes=new_classes_itter,
                               lr=lr,
                               batch_size=batch_size,
                               epochs=num_of_epochs,
                               use_pre_trained_weights=use_pt_weight,
                               use_batch_norm=use_bn,
                               use_subpopulation=use_subpopulation,
                               sensitive_feature=sf,
                               loss_function=criterion,
                               use_old_class_data=use_old_class_data)

    print(model)

    accuracy_per_train_dataset, \
    loss_all_classes, \
    testing_accuracy_per_class, \
    training_accuracy_per_class = model.training_(dataset=data_loader_AffectNet,
                                                  new_classes=new_classes,
                                                  new_classes_itter=new_classes_itter,
                                                  preprocessing=dataset_pre_process)

    plot_testing_accuracy_per_classes(testing_accuracy_per_class, classes_iter=2, classes=classes)
    plot_training_accuracy(accuracy_per_train_dataset, classes_iter=new_classes_itter, classes=classes)
    plot_model_training_loss(loss_all_classes, classes_iter=new_classes_itter, classes=classes)
    plot_old_classes_accuracy(model)
    plot_new_classes_testing_accuracy_per_sf(model=model, classes_iter=new_classes_itter, classes=classes, sf_group=sf)


def config_parse(args):
    """
    Check the command line arguments if are give correctly
    :param args: args, the argument from command line
    :return: None
    """
    if args.lr < 0:
        raise AssertionError('The --lr(learning rate) value cannot be a negative number')
    if args.num_classes <= 0:
        raise AssertionError('The --num_classes value cannot be a negative or zero number')
    if args.batch_size < 0:
        raise AssertionError('The --batch_size value cannot be a negative number')
    if args.epochs < 0:
        raise AssertionError('The --epochs value cannot be a negative number')
    if args.sf != 'race4' and args.sf != 'age':
        raise AssertionError('The --sf value must be race4 or age')


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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Learning without forgetting for facial emotion recognition with subpopulation shift option')
    parser.add_argument('--num_classes', default=2, help='Number of new classes introduced each time to the model',
                        type=int)
    parser.add_argument('--lr', default=0.0001, type=float, help='The learning rate in the training')
    parser.add_argument('--batch_size', default=32, type=int, help='The batch size')
    parser.add_argument('--epochs', default=8, type=int, help='The number of epochs that needed for the training of '
                                                              'each classes group')
    parser.add_argument('--use_batch_norm', default=False, type=str_to_bool,
                        help='Identify if we have batch normalization into '
                             'the Network model')
    parser.add_argument('--use_pt_weights', default=False, type=str_to_bool,
                        help='Identify if we use pretrained weights on the feature extractor')
    parser.add_argument('--use_sp', default=True, type=str_to_bool,
                        help='Identify if we use subpopulation in our model training and testing')
    parser.add_argument('--sf', default='race4', type=str,
                        help='Identify which class will be the sensitive feature (The 4 races or binary age group)\n'
                             'The value must be race4 or age')
    parser.add_argument('--old_class_data', default=False, type=str_to_bool,
                        help='Identify if add for each new tasks classes data from all the old classes in the training')

    args = parser.parse_args()
    config_parse(args)
    main(args)
