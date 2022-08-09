
## Requirements
Install Dependencies

- Pytorch: 1.11 (see more on the [pytorch link](https://pytorch.org/get-started/locally/))
- OpenCV:  ``` pip install opencv-contrib-python ```
- tqdm:  ```pip install tqdm```
- pandas: ```pip install pandas```
 
## Download Dataset
To download the Dataset must be requested from this link :
http://mohammadmahoor.com/affectnet/

and after placed it into the folder: Dataset/input/affectnetsample/, the created csv files for the dataset is placed into the dataset directory.
<br/>
But can be created and from the python kernel file of Create_AffectNet_csv.ipynb

## Preprocessing of the Dataset

Get on the link:
https://github.com/dchen236/FairFace 
</br>
And make a sensitive feature label annotation by the command: 
</br>
```
python3 predict.py --csv "labels.csv"
```
<br/>
With this we create the sensitive feature label, but you must download the models of the label detection
<br/>

# Training the model:
With the command of we train the model using pre-trained weights and without subpopulation shift for 32 epochs:
```
python main.py --num_classes 2 --lr 0.0001 --epochs 8 --batch_size 32 --use_pt_weights True --use_batch_norm False --use_sp False
```

With the command of we train the model using pre-trained weights, with batch normalization and without subpopulation shift for 32 epochs:
```
python main.py --num_classes 2 --lr 0.0001 --epochs 8 --batch_size 32 --use_pt_weights True --use_batch_norm True --use_sp False
```


With the command of we train the model using pre-trained weights and with subpopulation(as sensitive feature has been set the race4) shift for 32 epochs:
```
python main.py --num_classes 2 --lr 0.0001 --epochs 8 --batch_size 32 --use_pt_weights True --use_batch_norm False --use_sp True --sf race4
```

With the command of we train the model using pre-trained weights and with subpopulation(as sensitive feature has been set the binary age) shift for 32 epochs:
```
python main.py --num_classes 2 --lr 0.0001 --epochs 8 --batch_size 32 --use_pt_weights True --use_batch_norm False --use_sp True --sf age
```

# Get the Datasets Plots

In the '/Dataset/input/' directory has been set the csv files of FER-13, EMOTIC and KC+ for their labels and the race annotations
to can create the dataset plots like the report from 'Datasets_plots.ipynb', python kernel.

