# Training-a-Multi-Task-Model-for-CGDST
Training a Multi-Task Model for Classification and Grasp Detection of Surgical Tools using Transfer Learning

## Dataset
The entire dataset after labeling (used labelImg_OBB, availbale on github) is available in zip format.
This dataset is not preprocessed for training, so it should be preprocessed before training according to the instruction given in About file along with the dataset.
The dataset used for training the model with EfficientNetV2S as feature extractor is also given in the Dataset Folder.

## Model Training
The file Multi_task_CGD_model.ipynb could be used for training the model with any feature extractor.
We have used Kaggle for training our model.
By default EfficientNetV2S is set as feature extractor.

## Results
The results after training of model with EfficientNetV2S as feature extractor are given in the Results folder.

## Research Article
The corresponding research article can be found here: https://rdcu.be/didZY
