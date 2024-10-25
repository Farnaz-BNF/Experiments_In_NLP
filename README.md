## Overview

This repository provides notebooks and scripts for the course `Experiments in NLP`.

This Experiment is inspired from [Lauscher et al. (2020)](https://aclanthology.org/2020.emnlp-main.363/). They have shown an improvement in the performance of transfer learning by fine-tuning with few-shots from the target language. In their experimental setup, they fine-tune first with English as the source language and then, continue fine-tuning with few-shots in the target language. However, the impact of the order in which fine-tuning is applied has not been thoroughly investigated.

This experiment studies whether fine-tuning the model on the source language first, followed by few-shot instances from the target language, leads to different performance outcomes compared to fine-tuning with few-shot instances first and then, continuing fine-tuning on the source language. By focusing on the Cross-lingual Natural Language Inference (XNLI) task, this study seeks to determine whether changing the order of the fine-tuning impacts the model’s ability to perform cross-lingual inference.

----------
### Directories and Files

There are eight `python` scripts and one `python notebook file` and `requirements.txt` in this repository.

* `Main_High_level_task.ipynb`: This is the main file of my experiments code that used other scripts inside.

Also, there are two subfolders with the following structure:

1. `/Data`
   
    To run scripts without errors, you should put all data here. you can use first cell of `Main_High_level_task.ipynb` to get data.

2. `/Models`
   
    If you want to check pre-trained model you can download them and put them in this directory.

Note: Each directory has its own `README` file inside with more details.

------------
## Modules
To run scripts, you need to have the following modules:
```
os
re
csv
glob
nltk
tqdm
spacy
numpy
torch
pandas
pickle
random
logging
sklearn
seaborn
lang2vec
argparse
datasets
itertools
matplotlib
collections
transformers==2.3.0
```

---

## Environment

Environment that was used to run experiment is as follows:
1. Google Colab A100 GPU:
 Also, `requirements.txt` includes all modules that were available when running the experiment.  





