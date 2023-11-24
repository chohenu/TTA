This repo is follow by https://github.com/DianCh/AdaContrast

# CNA-TTA: Clean and Noisy Aware Feature Learning within Clusters for Online-Offline Test-Time Adaptation 

![Main figure](media/main.png)

## Installation
To use the repository, we provide a conda environment.
```bash
Make env
Make over
Make build
Make exec
```

## Prepare dataset
We download [VisDA-C](https://github.com/VisionLearningGroup/taskcv-2017-public/tree/master/classification), [DomainNet(cleaned version)](http://ai.bu.edu/M3SDA/), and [PACS](https://drive.google.com/drive/folders/0B6x7gtvErXgfUU1WcGY5SzdwZVk?resourcekey=0-2fvpQY_QSyJf2uIECzqPuQ).

To preprocess dataset, we follow same as in [AdaContrast](https://github.com/DianCh/AdaContrast).


## Source Training
To train a source model on each dataset, you can run bash files in src_${dataset} folder.



## Adpatation for Target
For adaptation for target, you can run bash files in src_${dataset} folder.