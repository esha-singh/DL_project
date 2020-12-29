# DL_project

Repository for CSCI 8980/5980 Deep Learning Project on Google Landmark Recognition Challenge 2020. 

## Data preparation
Download dataset from [Kaggle Google Landmark Recognition 2020](https://www.kaggle.com/c/landmark-recognition-2020/data)
and extract to this root directory.
 
## Training
Train the DELG model with only using global features
```
python train.py --config-file configs/delg.yaml
```

## Testing
Download the pretrained model from [here](https://drive.google.com/drive/folders/1tGtt8-wYba21Wwf-rWtsJQrmhHP79qCU?usp=sharing)

The csv file of the test set is 'sample_submission.csv', which is from [Kaggle Google Landmark Recognition 2020](https://www.kaggle.com/c/landmark-recognition-2020/data)

For prediction, we use the Kaggle provided train set as the gallery set for image retrieval given an query image (from test set).

For testing, run:
```
python test.py --config-file configs/delg.yaml --save-gallery-feats
```

## Results
Using only global features for landmarks recognition get the GAP score of 0.355 and accuracy of 61.3% (with image resolution of 256x256)
Using local features for re-ranking has 17% of improvement in GAP score.
