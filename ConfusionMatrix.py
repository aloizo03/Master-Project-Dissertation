from sklearn.metrics import multilabel_confusion_matrix


class ConfusionMatrix:
    """
    This class create a confusion metric based the predictions and the ground truth labels
    Can create CM for multi-classes
    """
    def __init__(self, predicts, ground_truth_labels, classes):
        self.predicts = predicts
        self.ground_truth_labels = ground_truth_labels
        self.classes = classes

        m_l_CM = multilabel_confusion_matrix(ground_truth_labels, predicts)
        # For each class put TP, TN, FP, FN into a matrix
        self.TP = []
        self.TN = []
        self.FP = []
        self.FN = []
        for matrix in m_l_CM:
            self.TP.append(matrix[0, 0])
            self.FN.append(matrix[0, 1])
            self.TN.append(matrix[1, 0])
            self.FP.append(matrix[1, 1])

    def get_TP_per_class(self, class_index):
        return self.TP[class_index]

    def get_TN_per_class(self, class_index):
        return self.TN[class_index]

    def get_FP_per_class(self, class_index):
        return self.FP[class_index]

    def get_FN_per_class(self, class_index):
        return self.FN[class_index]

    def calculate_accuracy_all_classes(self):
        acc = (self.predicts == self.ground_truth_labels).sum()
        return acc / len(self.predicts)

    def calculate_TPR(self, class_index):
        return self.TP[class_index] / (self.TP[class_index] + self.FN[class_index])

    def calculate_FPR(self, class_index):
        return self.FP[class_index] / (self.FP[class_index] + self.TN[class_index])

    def calculate_TNR(self, class_index):
        return self.FN[class_index] / (self.FP[class_index] + self.TN[class_index])

    def calculate_FNR(self, class_index):
        return 1 - self.calculate_TPR(class_index)

    def calculate_TPR_for_all_classes(self):
        TPR = 0
        for i in range(len(self.classes)):
            TPR += self.calculate_TPR(i)
        return TPR / len(self.classes)

    def calculate_FPR_for_all_classes(self):
        FPR = 0
        for i in range(len(self.classes)):
            FPR += self.calculate_FPR(i)
        return FPR / len(self.classes)

    def calculate_TNR_for_all_classes(self):
        TNR = 0
        for i in range(len(self.classes)):
            TNR += self.calculate_TNR(i)
        return TNR / len(self.classes)

    def calculate_FNR_for_all_classes(self):
        FNR = 0
        for i in range(len(self.classes)):
            FNR += self.calculate_FNR(i)
        return FNR / len(self.classes)

    def calculate_balance_accuracy_class(self, class_index):
        TP = self.get_TP_per_class(class_index)
        TN = self.get_TN_per_class(class_index)
        FN = self.get_FP_per_class(class_index)
        FP = self.get_FP_per_class(class_index)

        return (TP + TN) / (TP + FN + FP + TN)

    def calculate_balance_accuracy_for_all_classes(self):
        TP = 0
        TN = 0
        FN = 0
        FP = 0
        for i in range(len(self.classes)):
            TP += self.get_TP_per_class(i)
            TN += self.get_TN_per_class(i)
            FN += self.get_FP_per_class(i)
            FP += self.get_FP_per_class(i)

        #         TP = TP/len(self.classes)
        #         TN = TN/len(self.classes)
        #         FN = FN/len(self.classes)
        #         FP = FP/len(self.classes)

        return (self.calculate_TPR_for_all_classes() + self.calculate_TNR_for_all_classes()) / 2

    def calculate_accuracy_class(self, class_index):
        indexes = [index for index in range(len(self.ground_truth_labels)) if
                   class_index == self.ground_truth_labels[index]]
        predicts = [self.predicts[i] for i in indexes]
        labels = [self.ground_truth_labels[i] for i in indexes]

        acc = (predicts == labels).sum()
        return acc / len(labels)
