import matplotlib.pyplot as plt
import numpy as np
import matplotlib.colors as mcolors

races4 = {'Asian', 'Black', 'Indian', 'White'}
non_white_races4 = {'Black', 'Indian'}
white_races4 = {'Asian', 'White'}


def plot_balance_accuracy_per_sf(CM_list):
    colors = mcolors.TABLEAU_COLORS
    for cm in CM_list:
        axis_x_labels = []
        balance_acc = []
        for sf, cm_per_sm in cm.items():
            axis_x_labels.append(sf)
            balance_acc.append(cm_per_sm.calculate_balance_accuracy_for_all_classes())
        plt.figure(figsize=(10, 8), dpi=70)
        plt.ylim([0, 1])
        plt.title('Balance accuracy per sensitive feature')
        plt.bar(axis_x_labels, balance_acc, color=colors)
        plt.show()


def plot_model_training_loss(loss_per_training_class,
                             classes=['anger', 'contempt', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise'],
                             classes_iter=2):
    plt.figure(figsize=(9, 3))
    plt.title('Training model loss for old classes and new classes')
    for loss, classes_per_tt in zip(loss_per_training_class, zip(*[iter(classes)] * classes_iter)):
        plt.plot(loss, label=f'Loss for classes {classes_per_tt}')
        plt.legend(loc="best")
    plt.show()


def plot_training_accuracy(acc_per_training_class,
                           classes=['anger', 'contempt', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise'],
                           classes_iter=2):
    plt.figure(figsize=(9, 3))
    markers = ['.', ',', 'o', 'v', '^', '<', '>', '1', '2', '3', '4', '8', 's', 'p', 'P', '*', 'h', 'H', '+', 'x', 'X',
               'D', 'd', '|', '_']
    plt.title('Training accuracy for old classes and new classes')
    i = 0
    for acc, classes_per_tt in zip(acc_per_training_class, zip(*[iter(classes)] * classes_iter)):
        plt.plot(acc, marker=markers[i], label=f'Accuracy for classes {classes_per_tt}')
        plt.legend(loc="best")
        i += 1
    plt.show()


def plot_testing_accuracy_per_classes(accuracy_per_classes,
                                      classes=['anger', 'contempt', 'disgust', 'fear', 'happy', 'neutral', 'sad',
                                               'surprise'],
                                      classes_iter=2):
    x_axis_label = [c for c in zip(*[iter(classes)] * classes_iter)]
    x = [i for i, c in enumerate(zip(*[iter(classes)] * classes_iter))]
    plt.figure(figsize=(9, 3))
    plt.title('Testing accuracy for LwF model\nFor old classes and new classes')
    plt.plot(accuracy_per_classes, marker='o')
    plt.xticks(x, x_axis_label)
    plt.show()


def plot_old_classes_accuracy(model):
    fig, axs = plt.subplots(len(model.testing_accuracy_per_classes_update), 1)
    fig.tight_layout()
    colors = mcolors.TABLEAU_COLORS
    for i, accuracy_per_new_task in enumerate(model.testing_accuracy_per_classes_update):
        accuracy_lst = []
        emotion_lst = []
        for emotion, predictions in accuracy_per_new_task.items():
            accuracy = (np.array(predictions['predict']) == np.array(predictions['label'])).sum() / len(
                predictions['label'])
            accuracy_lst.append(accuracy)
            emotion_lst.append(emotion)

        axs[i].bar(emotion_lst, accuracy_lst, color=colors)
    plt.show()


def plot_new_classes_testing_accuracy(model,
                                       classes=['anger', 'contempt', 'disgust', 'fear', 'sad', 'surprise', 'happy',
                                                'neutral'],
                                       classes_iter=2):
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

        axs[i].bar(emotion_lst, accuracy_lst, color=colors)
        i += 1
    plt.show()


##################################################################

def plot_test_accuracy_per_sf(model):
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
        plt.figure(figsize=(10, 8), dpi=70)
        plt.title('Testing accuracy per Class and Race')
        plt.barh(axis_x_labels, acc, color=colors)
        plt.show()


def plot_new_classes_testing_accuracy_per_sf(model,
                                             classes=['anger', 'contempt', 'disgust', 'fear', 'sad', 'surprise',
                                                      'happy', 'neutral'],
                                             classes_iter=2):
    colors = mcolors.TABLEAU_COLORS
    for accuracy_per_new_task, classes_per_tt in zip(model.testing_accuracy_per_classes_update,
                                                     zip(*[iter(classes)] * classes_iter)):
        axis_x_labels = []
        acc = []
        for emotion in classes_per_tt:
            predictions = accuracy_per_new_task[emotion]
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
        plt.figure(figsize=(10, 8), dpi=70)
        plt.title('Testing accuracy per Class and Race')
        plt.barh(axis_x_labels, acc, color=colors)
        plt.show()


def plot_balance_accuracy_per_sf(CM_list):
    colors = mcolors.TABLEAU_COLORS
    for cm in CM_list:
        axis_x_labels = []
        balance_acc = []
        for sf, cm_per_sm in cm.items():
            axis_x_labels.append(sf)
            balance_acc.append(cm_per_sm.calculate_balance_accuracy_for_all_classes())
        plt.figure(figsize=(10, 8), dpi=70)
        plt.ylim([0, 1])
        plt.title('Balance accuracy per sensitive feature')
        plt.bar(axis_x_labels, balance_acc, color=colors)
        plt.show()
