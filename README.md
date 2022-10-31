# Improving Loss Function for Deep CNN-based AIA
This is the official PyTorch implementation of the paper "[Improving Loss Function for Deep CNN-based Automatic Image Annotation]()"

## Datasets
There are three well-known datasets that are mostly used in AIA tasks. The table below provides details about these datasets. It is also possible to download them by the given links. (After downloading each dataset, replace its 'images' folder with the corresponding 'images' folder in the 'datasets' folder).

| *Dataset* | *Num of images* | *Num of training images* | *Num of testing images*  | *Num of vocabularies*  | *Labels per image*  | *Image per label* |
| :------------: | :-------------: | :-------------: | :-------------: | :------------: | :-------------: | :-------------: |
| [Corel 5k](https://www.kaggle.com/datasets/parhamsalar/corel5k) | 5,000 | 4,500 | 500 | 260 | 3.4 | 58.6 |
| [IAPR TC-12](https://www.kaggle.com/datasets/parhamsalar/iaprtc12) | 19,627 | 17,665 | 1962 | 291 | 5.7 | 347.7 |
| [ESP Game](https://www.kaggle.com/datasets/parhamsalar/espgame) | 20,770 | 18,689 | 2081 | 268 | 4.7 | 362.7 |

## Convolutional model
**TResNet-M**
![TResNet-M](https://user-images.githubusercontent.com/85555218/198952123-391fdfe0-4bd2-4129-982c-c1074279b099.png)

## Train and Evaluation
To train the model in Spyder IDE use the code below:
```python
run main.py --data {select training dataset} --loss-function {select loss function}
```
Please note that:
1) You should put **Corel-5k**, **ESP-Game** or **IAPR-TC-12** in {select training dataset}.

2) You should put the **proposedLoss** in {select loss function}.

To evaluate the model in Spyder IDE use the code below:
```python
run main.py --data {select training dataset} --loss-function {select loss function} --evaluate
```

## Results
Proposed method:
| data | precision | recall | f1-score | N+ |
| :------------: | :------------: | :------------: | :------------: | :------------: |
Corel 5k | 0.466 | 0.554 | **0.506** | **189** |
IAPR TC-12 | 0.503 | 0.562 | **0.531** | **285** |
ESP Game | 0.423 | 0.484 | **0.452** | **261** |

Proposed method + MCC:
| data | precision | recall | f1-score | N+ |
| :------------: | :------------: | :------------: | :------------: | :------------: |
Corel 5k | 0.484 | 0.563 | **0.520** | **191** |
IAPR TC-12 | 0.562 | 0.515 | **0.537** | **277** |
ESP Game | 0.508 | 0.421 | **0.461** | **255** |

## Citation

## Contact
I would be happy to answer any questions you may have - Ali Salar (parham1998resume@gmail.com)
