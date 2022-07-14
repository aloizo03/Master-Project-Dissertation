from Dataset import DataLoader_Affect_Net
from Model import Model, Model_with_LwF
from dataset_preprocessing import *
from Plots import *
from config import *
import argparse


def main(args):
    new_classes_itter = args.num_classes
    batch_size = args.batch_size
    lr = args.lr
    num_of_epochs = args.epochs
    use_pt_weight = args.use_pt_weights
    use_bn = args.use_batch_norm
    use_subpopulation = args.use_sp

    classes = emotions_classes
    old_classes = [classes[i] for i in range(new_classes_itter)]
    new_classes = [classes[i] for i in range(new_classes_itter, len(emotions_classes))]

    dataset_pre_process = pre_processing()
    data_loader_AffectNet = DataLoader_Affect_Net(classes=old_classes,
                                                  pre_processing=dataset_pre_process,
                                                  use_subpopulation=use_subpopulation)

    model_LwF = Model_with_LwF(classes=old_classes,
                               total_classes=new_classes_itter,
                               lr=lr,
                               batch_size=batch_size,
                               epochs=num_of_epochs,
                               use_pre_trained_weights=use_pt_weight,
                               use_batch_norm=use_bn,
                               use_subpopulation=use_subpopulation)
    accuracy_per_train_dataset, \
    loss_all_classes, \
    testing_accuracy_per_class, \
    training_accuracy_per_class = model_LwF.training_(dataset=data_loader_AffectNet,
                                                      new_classes=new_classes,
                                                      new_classes_itter=new_classes_itter,
                                                      preprocessing=dataset_pre_process)

    plot_testing_accuracy_per_classes(testing_accuracy_per_class, classes_iter=2, classes=classes)
    plot_training_accuracy(accuracy_per_train_dataset, classes_iter=new_classes_itter, classes=classes)
    plot_model_training_loss(loss_all_classes, classes_iter=new_classes_itter, classes=classes)
    plot_old_classes_accuracy(model_LwF)
    plot_test_accuracy_per_sf(model=model_LwF)


def config_parse(args):
    if args.lr < 0:
        raise AssertionError('The --lr(learning rate) value cannot be a negative number')
    if args.num_classes < 0:
        raise AssertionError('The --num_classes value cannot be a negative number')
    if args.batch_size < 0:
        raise AssertionError('The --batch_size value cannot be a negative number')
    if args.epochs < 0:
        raise AssertionError('The --epochs value cannot be a negative number')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Learning without forgetting for facial emotion recognition')
    parser.add_argument('--num_classes', default=2, help='Number of new classes introduced each time to the model',
                        type=int)
    parser.add_argument('--lr', default=0.0001, type=float, help='The learning rate in the training')
    parser.add_argument('--batch_size', default=8, type=int, help='The batch size')
    parser.add_argument('--epochs', default=5, type=int, help='The number of epochs that needed for the training of '
                                                              'each classes group')
    parser.add_argument('--use_batch_norm', default=False, type=bool,
                        help='Identify if we have batch normalization into '
                             'the Network model')
    parser.add_argument('--use_pt_weights', default=False, type=bool,
                        help='Identify if we use pretrained weights on the feature extractor')
    parser.add_argument('--use_sp', default=True, type=bool,
                        help='Identify if we use subpopulation in our model training and testing')

    args = parser.parse_args()
    config_parse(args)
    main(args)
