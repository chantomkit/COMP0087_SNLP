# COMP0087_SNLP
NLPeekaboo

---
### Building pretraining data from BabyLM Challenge and TinyStories

Data download links:
- BabyLM: https://babylm.github.io/
- TinyStories: https://www.kaggle.com/datasets/thedevastator/tinystories-narrative-classification

### Usage

1. Download the data and follow the directory structure below

(Default) Directory Structure:
```
│ 
├── data/ (Data sources)
│   └── babylm_data/
│       ├── babylm_10M/
│           └── *.train
│       ├── babylm_100M/
│           └── *.train
│       └── .../
│   └── tinystories_data/
│       ├── train.csv
│       └── validation.csv
├── processed_data/ (Processed data storage)
│   └── *.pkl
├── pretraining/ (Pretraining data processing functions)
│   ├── data_pipeline.py
│   └── data_utils.py
└── run_pretraining_data.py (Runner file)
```

2. Do `python run_pretraining_data.py` without any arguments will run the pipeline with default settings.<br>
To summarize the default behavior, the pipeline will process the full corpus of both BabyLM and TinyStories following the default directory sturcture. It will use the model `SamLowe/roberta-base-go_emotions` with cpu for emotion inference, then it will further filter out non-confident emotions predictions and neutral emotion.<br>
To see what each argument does, do `python run_pretraining_data.py -h`.
---