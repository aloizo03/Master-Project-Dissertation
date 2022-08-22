# Dataset configurations

To can access into the dataset must unzip on this directory the train_set.tar and the val_set.tar when the permition is given from the AffectNet dataset. Their folder name
must not be changed as the program executed with this folder names.
</br>

The FairFace annotation tool are included nad is the predict.py file, which has been change to 
can be used from the dataset csv.

The annotation of a csv file can be executed with the command:
```
python .\predict.py --csv ".\labels.csv" --out "test_outputs.csv"
```

On the main directory on the config.py must change the data_filename and race_labels_filename if their 
filename are changed. The same with the validation csv file name (val_data_filename and val_race_labels_filename).