from sklearn.metrics import multilabel_confusion_matrix


class ConfusionMatrix:

    def __init__(self, predicts, ground_truth_labels, classes):
        self.predicts = predicts
        self.ground_truth_labels = ground_truth_labels
        self.classes = classes

        m_l_CM = multilabel_confusion_matrix(ground_truth_labels, predicts, classes)
        # For each class put TP, TN, FP, FN into a matrix
        self.TP = []
        self.TN = []
        self.FP = []
        self.FN = []
        print(m_l_CM)
        for matrix in m_l_CM:
            self.TP.append(matrix[0, 0])
            self.FN.append(matrix[0, 1])
            self.TN.append(matrix[1, 0])
            self.FP.append(matrix[1, 1])

    def calculate_accuracy_all_classes(self):
        acc = (self.predicts == self.ground_truth_labels).sum()
        return acc / len(self.predicts)

    def calculate_TPR(self, class_index):
        return self.TP[class_index]/(self.TP[class_index] + self.FN[class_index])

    def calculate_FPR(self, class_index):
        return self.FP[class_index]/(self.FP[class_index] + self.TN[class_index])

    def calculate_TNR(self, class_index):
        return self.FN[class_index]/(self.FP[class_index] + self.TN[class_index])

    def calculate_FNR(self, class_index):
        return 1 - self.calculate_TPR(class_index)

    def calculate_accuracy_class(self, class_index):
        indexes = [index for index in range(len(self.ground_truth_labels)) if class_index == self.ground_truth_labels[index]]
        predicts = [self.predicts[i] for i in indexes]
        labels = [self.ground_truth_labels[i] for i in indexes]

        acc = (predicts == labels).sum()
        return acc / len(labels)
