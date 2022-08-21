import matplotlib.pyplot as plt
import numpy as np
import matplotlib.colors as mcolors
from ConfusionMatrix import ConfusionMatrix
import cv2

plt.rcParams.update({'font.size': 14})

races4 = {'Asian', 'Black', 'Indian', 'White'}
non_white_races4 = {'Black', 'Indian'}
white_races4 = {'Asian', 'White'}

binary_age_groups = {'young', 'old'}


def plot_balance_accuracy_per_sf(CM_list):
    """
    Getting the confusion matrix create the Balance accuracy bet sf group
    :param CM_list: list, a list with the Confusion matrix
    :return: None
    """
    colors = mcolors.TABLEAU_COLORS
    for cm in CM_list:
        axis_x_labels = []
        balance_acc = []
        for sf, cm_per_sm in cm.items():
            axis_x_labels.append(sf)
            balance_acc.append(cm_per_sm.calculate_balance_accuracy_for_all_classes())
        plt.figure(figsize=(10, 8), dpi=110)
        plt.ylim([0, 1])
        plt.title('Balance accuracy per sensitive feature')
        plt.bar(axis_x_labels, balance_acc, color=colors)
        plt.show()


def plot_model_training_loss(loss_per_training_class,
                             classes=['anger', 'sad', 'disgust', 'fear', 'contempt', 'surprise', 'happy', 'neutral'],
                             classes_iter=2):
    """
    This function create the plot fro the model loss for each tasks pair training
    :param loss_per_training_class: list, a list of lists with the training model loss
    :param classes: list, a list of list which each list have the total model loss for each training class pair
    :param classes_iter: int, the number of class addition for each time of training the LwF
    :return: None
    """
    plt.figure(figsize=(9, 3))
    plt.title('Training model loss for old classes and new classes')
    markers = ['.', ',', 'o', 'v', '^', '<', '>', '1', '2', '3', '4', '8', 's', 'p', 'P', '*', 'h', 'H', '+', 'x', 'X',
               'D', 'd', '|', '_']
    i = 0
    for loss, classes_per_tt in zip(loss_per_training_class, zip(*[iter(classes)] * classes_iter)):
        plt.plot(range(1, len(loss) + 1), loss, marker=markers[i], label=f'Loss for classes {classes_per_tt}')
        plt.legend(loc="best")
        plt.xticks(range(1, len(loss) + 1))
        i += 1
    plt.show()


def plot_training_accuracy(acc_per_training_class,
                           classes=['anger', 'sad', 'disgust', 'fear', 'contempt', 'surprise', 'happy', 'neutral'],
                           classes_iter=2):
    """
    This function create a plot with the training accuracy for each epoch
    :param acc_per_training_class: list, a list of list which each list have the training accuracy for each epoch
    :param classes: list, a list of list which each list have the total model loss for each training class pair
    :param classes_iter: int, the number of class addition for each time of training the LwF
    :return: None
    """
    plt.figure(figsize=(9, 3))
    markers = ['.', ',', 'o', 'v', '^', '<', '>', '1', '2', '3', '4', '8', 's', 'p', 'P', '*', 'h', 'H', '+', 'x', 'X',
               'D', 'd', '|', '_']
    plt.title('Training accuracy for old classes and new classes')
    i = 0
    for acc, classes_per_tt in zip(acc_per_training_class, zip(*[iter(classes)] * classes_iter)):
        plt.plot(range(1, len(acc) + 1), acc, marker=markers[i], label=f'Accuracy for classes {classes_per_tt}')
        plt.xticks(range(1, len(acc) + 1))
        plt.yticks(np.arange(0, 1, step=0.15))
        plt.legend(loc="best")
        i += 1
    plt.show()


def plot_testing_accuracy_per_classes(accuracy_per_classes,
                                      classes=['anger', 'sad', 'disgust', 'fear', 'contempt', 'surprise', 'happy',
                                               'neutral'],
                                      classes_iter=2):
    """
    This function create a plot for the testing accuracy for only the training classes pair (Doesn't include the old classes)
    :param accuracy_per_classes: list, a list with the testing accuracy for each training classes pair
    :param classes: list, a list of list which each list have the total model loss for each training class pair
    :param classes_iter: int, the number of class addition for each time of training the LwF
    :return: None
    """
    x_axis_label = [c for c in zip(*[iter(classes)] * classes_iter)]
    x = [i for i, c in enumerate(zip(*[iter(classes)] * classes_iter))]
    plt.figure(figsize=(9, 3))
    plt.title('Testing accuracy for LwF model\nFor old classes and new classes')
    plt.plot(accuracy_per_classes, marker='o')
    plt.xticks(x, x_axis_label)
    plt.show()


def plot_old_classes_accuracy(model):
    """
    This function create a plot for each training pair all of the classes testing accuracy
    :param model: Model, the model where was implemented the training and the testing
    :return: None
    """
    fig, axs = plt.subplots(len(model.testing_accuracy_per_classes_update), 1)
    fig.tight_layout()
    colors = mcolors.TABLEAU_COLORS
    for i, accuracy_per_new_task in enumerate(model.testing_accuracy_per_classes_update):
        accuracy_lst = []
        emotion_lst = []
        # For each emotion class takes their predictions and calculate the accuracy
        for emotion, predictions in accuracy_per_new_task.items():
            accuracy = (np.array(predictions['predict']) == np.array(predictions['label'])).sum() / len(
                predictions['label'])
            accuracy_lst.append(accuracy)
            emotion_lst.append(emotion)

        axs[i].bar(emotion_lst, accuracy_lst, color=colors)
        axs[i].set_yticks(np.arange(0, 1, step=0.15))
        axs[i].set_ylim([0, 1])
    plt.show()


def plot_new_classes_testing_accuracy(model,
                                      classes=['anger', 'sad', 'disgust', 'fear', 'contempt', 'surprise', 'happy',
                                               'neutral'],
                                      classes_iter=2):
    """
    This function create a plot for testing for all the classes in each training time
    :param model: Model, the model where was implemented the training and the testing
    :param classes: list, a list of list which each list have the total model loss for each training class pair
    :param classes_iter: int, the number of class addition for each time of training the LwF
    :return: None
    """
    fig, axs = plt.subplots(len(model.testing_accuracy_per_classes_update), 1)
    fig.tight_layout()
    colors = mcolors.TABLEAU_COLORS
    i = 0
    for accuracy_per_new_task, classes_per_tt in zip(model.testing_accuracy_per_classes_update,
                                                     zip(*[iter(classes)] * classes_iter)):
        accuracy_lst = []
        emotion_lst = []
        for emotion in classes_per_tt:
            predictions = accuracy_per_new_task[emotion]
            accuracy = (np.array(predictions['predict']) == np.array(predictions['label'])).sum() / len(
                predictions['label'])
            accuracy_lst.append(accuracy)
            emotion_lst.append(emotion)

        print(accuracy_lst)
        axs[i].bar(emotion_lst, accuracy_lst, color=colors)
        axs[i].set_xticks(np.arange(0, 1, step=0.15))
        i += 1
    plt.show()


def plot_test_accuracy_per_sf(model):
    """
    This function create a plot for accuracy for each class and each sensitive feature
    :param model: Model, the model where was implemented the training and the testing
    :return: None
    """
    colors = mcolors.TABLEAU_COLORS

    for i, accuracy_per_new_task in enumerate(model.testing_accuracy_per_classes_update):
        axis_x_labels = []
        acc = []
        for emotion, predictions in accuracy_per_new_task.items():
            # sensitive_features = list(set(predictions['sf']))
            sensitive_features = list(set(predictions['sf']))
            preds_lst = predictions['predict']
            labels_lst = predictions['label']
            sf_lst = predictions['sf']
            for sf in sensitive_features:
                indexes = [index for index in range(len(sf_lst)) if sf == sf_lst[index]]
                predicts = [preds_lst[i] for i in indexes]
                labels = [labels_lst[i] for i in indexes]
                acc.append((np.array(predicts) == np.array(labels)).sum() / len(predicts))
                axis_x_labels.append(f'{emotion}\n{sf}')
        plt.figure(figsize=(10, 8), dpi=110)
        plt.title('Testing accuracy per Class and Race')
        plt.barh(axis_x_labels, acc, color=colors)
        plt.xticks(np.arange(0, 1, step=0.1))
        plt.show()


def plot_new_classes_testing_accuracy_per_sf(model,
                                             classes=['anger', 'sad', 'disgust', 'fear', 'contempt', 'surprise',
                                                      'happy', 'neutral'],
                                             classes_iter=2,
                                             sf_group='race4'):
    """
    This function create a testing plot for only the new classes and each sensitive feature
    :param sf_group:
    :param model: Model, the model where was implemented the training and the testing
    :param classes: list, a list of list which each list have the total model loss for each training class pair
    :param classes_iter: int, the number of class addition for each time of training the LwF
    :return: None
    """
    colors = mcolors.TABLEAU_COLORS
    for accuracy_per_new_task, classes_per_tt in zip(model.testing_accuracy_per_classes_update,
                                                     zip(*[iter(classes)] * classes_iter)):
        axis_x_labels = []
        acc = []
        # For each emotion from the new classes
        for emotion in classes_per_tt:
            predictions = accuracy_per_new_task[emotion]
            if sf_group == 'race4':
                sensitive_features = races4
            elif sf_group == 'age':
                sensitive_features = binary_age_groups
            preds_lst = predictions['predict']
            labels_lst = predictions['label']
            sf_lst = predictions['sf']

            # takes all the sensitive features
            for sf in sensitive_features:
                indexes = [index for index in range(len(sf_lst)) if sf == sf_lst[index]]
                predicts = [preds_lst[i] for i in indexes]
                labels = [labels_lst[i] for i in indexes]
                acc.append((np.array(predicts) == np.array(labels)).sum() / len(predicts))
                axis_x_labels.append(f'{emotion}\n{sf}')
        plt.figure(figsize=(10, 8), dpi=120)
        plt.title('Testing accuracy per Class and Race')
        plt.barh(axis_x_labels, acc, color=colors)
        plt.xticks(np.arange(0, 1, step=0.1))
        plt.show()


def plot_balance_accuracy_per_sf(CM_list,
                                 classes=['anger', 'sad', 'disgust', 'fear', 'contempt', 'surprise', 'happy',
                                          'neutral'],
                                 classes_iter=2):
    """
    This function plot the balance accuracy for each sensitive feature
    :param CM_list: list, a list with the Confusion matrix
    :param classes: list, a list of list which each list have the total model loss for each training class pair
    :param classes_iter: int, the number of class addition for each time of training the LwF
    :return: None
    """
    colors = mcolors.TABLEAU_COLORS
    for cm, classes_per_tt in zip(CM_list, zip(*[iter(classes)] * classes_iter)):
        axis_x_labels = []
        balance_acc = []
        for sf, cm_per_sm in cm.items():
            axis_x_labels.append(sf)
            balance_acc.append(cm_per_sm.calculate_balance_accuracy_for_all_classes())
        plt.figure(figsize=(10, 8), dpi=70)
        plt.ylim(np.arange(0, 1, step=0.1))
        plt.title(f'Balance accuracy per sensitive feature for classes {classes_per_tt}')
        plt.bar(axis_x_labels, balance_acc, color=colors)
        plt.show()


def calculate_confunsion_matrix_per_sf_only_new_classes(model,
                                                        classes=['anger', 'sad', 'disgust', 'fear', 'contempt',
                                                                 'surprise', 'happy', 'neutral'],
                                                        classes_iter=2):
    """
    This function plot the balance accuracy for each sensitive feature for each new classes pair
    :param CM_list: list, a list with the Confusion matrix
    :param classes: list, a list of list which each list have the total model loss for each training class pair
    :param classes_iter: int, the number of class addition for each time of training the LwF
    :return: None
    """
    cm_list = []
    for accuracy_per_new_task, classes_per_tt in zip(model.testing_accuracy_per_classes_update,
                                                     zip(*[iter(classes)] * classes_iter)):
        classes_per_tt = list(classes_per_tt)
        labels_lst = []
        preds_lst = []
        sf_lst = []

        sensitive_features = list(races4)
        for emotion in classes_per_tt:
            predictions = accuracy_per_new_task[emotion]
            preds_lst.extend(predictions['predict'])
            labels_lst.extend(predictions['label'])
            sf_lst.extend(predictions['sf'])

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


def plot_image_and_labels(image_with_labels_lst,
                          sf='race4'):
    """
    Taken a list with dictionary make a plot for each class pair with the image the emotion label the emotion which
    predicted from the model and the sensitive feature group.
    :param image_with_labels_lst: list, a list with all the image information
    :param sf: str, the sensitive feature group can be race4 or age
    :return: None
    """
    for image_with_labels_dict in image_with_labels_lst:
        if sf == 'race4':
            lst = image_with_labels_dict[list(races4)[0]]
            lst = lst['images']
            fig, axs = plt.subplots(len(races4), len(lst), figsize=(5, 5))
        elif sf == 'age':
            lst = image_with_labels_dict[list(binary_age_groups)[0]]
            lst = lst['images']
            fig, axs = plt.subplots(len(binary_age_groups), len(lst), figsize=(5, 5))

        fig.set_size_inches(18.5, 10.5, forward=True)
        fig.set_dpi(80)
        for i, domain in enumerate(image_with_labels_dict.keys()):
            labels_dict = image_with_labels_dict[domain]
            img_lst = labels_dict['images']
            labels_lst = labels_dict['labels']
            predictions_lst = labels_dict['predictions']
            sf_lst = labels_dict['sf']
            for j in range(len(img_lst)):
                img = cv2.imread(img_lst[j])
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                axs[i, j].imshow(img_rgb)
                axs[i, j].set_xlabel(f"label : {labels_lst[j]}\n"
                                     f"prediction : {predictions_lst[j]}\n"
                                     f"Domain : {sf_lst[j]}")
                axs[i, j].set_xticks(())
                axs[i, j].set_yticks(())
        plt.subplots_adjust(left=0.1,
                            bottom=0.1,
                            right=0.9,
                            top=0.9,
                            wspace=0.4,
                            hspace=0.4)
        plt.show()
