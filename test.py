from Dataset import DataLoader_Affect_Net

old_classes = ['anger', 'contempt']
new_classes = [ 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
data_loader_AffectNet = DataLoader_Affect_Net(classes=old_classes)
test_set = set(data_loader_AffectNet.labels)
print(test_set)
