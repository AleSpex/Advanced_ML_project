# Advanced Machine Learning Project - Domain-To-Text 

Basic code to reproduce the baselines (point 1 of the project). 

## Dataset

1 - Download PACS dataset from the portal of the course in the "project_topics" folder.

2 - Place the dataset in the DomainToText_AMLProject folder making sure that the images are organized in this way:

```
PACS/kfold/art_painting/dog/pic_001.jpg
PACS/kfold/art_painting/dog/pic_002.jpg
PACS/kfold/art_painting/dog/pic_003.jpg
...
```

## Pretrained models

In order to reproduce the values reported in the table, you have to download the pretrained models from this link: https://drive.google.com/drive/folders/17tWDDDPY9fRLrnL3YbwkHrilq12oii2M?usp=sharing

Then, you have to put the "outputs" folder into 

```
/DomainToText_AMLProject/
```


## Environment

To run the code you have to install all the required libraries listed in the "requirements.txt" file.

For example, if you read

```
torch==1.4.0
```

you have to execute the command:

```
pip install torch==1.4.0
```

