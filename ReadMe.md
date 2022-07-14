
## Requirements
Install Dependencies

- Pytorch: 1.11 (see more on the [pytorch link](https://pytorch.org/get-started/locally/))
- OpenCV:  ``` pip install opencv-contrib-python ```
- tqdm:  ```pip install tqdm```
- pandas: ```pip install pandas```
 
## Download Dataset
To download the Dataset must be downloaded from this link :
https://www.kaggle.com/datasets/tom99763/affectnethq?resource=download

and after placed it inot the folder: Dataset/input/affectnetsample/

## Preprocessing of the Dataset

Get on the link:
https://github.com/dchen236/FairFace 
</br>
And make a sensitive feature label annotation by the command: 
</br>
```
python3 predict.py --csv "labels.csv"
```
With this we create the sensitive feature label, but you must download the models of the label detection

# Training the model:
With the command of we train the model using pre-trained weights and subpopulation shift for 8 epochs:
```
python main.py --num_classes 2 --lr 0.0001 --epochs 8 --batch_size 12 --use_pt_weights True --use_batch_norm False --use_sp True
```
