# COMP0087_SNLP NLPeekaboo
---

## Building and augment corpus from BabyLM Challenge data
Data download links:
- BabyLM: https://babylm.github.io/

### Usage

1. Download the data and follow the directory structure below\
\
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
├── processed_data/ (Processed data storage)
│   └── *.pkl
├── babylm_augment/ (Augmented data storage)
│   └── *.pkl
├── babylm_pretrain_corpus/ (Corpus and metadata log storage)
│   ├── *.pkl
│   └── *.txt
├── pretraining/ (Helper functions packages)
│   ├── data_pipeline.py
│   └── data_utils.py
├── process_pretrain_data.py (Process BabyLM data runner file)
├── pretrain_data_augment.ipynb (Notebook for augmenting processed data)
└── pretrain_data_corpus.ipynb (Notebook for building final corpus)
```

2. Do `python process_pretrain_data.py` without any arguments will run the pipeline with default settings.\
\
To summarize the default behavior, the pipeline will process the full corpus of BabyLM following the default directory sturcture. It will use the model `SamLowe/roberta-base-go_emotions` with cpu for emotion inference, then it will further filter out non-confident emotions predictions and neutral emotion.\
\
To see what each argument does, do `python run_pretrain_data.py -h`.

3. Upon successful run of `process_pretrain_data.py`, there will be new directory storing pickle files of processed data in `processed_data/`

4. Download the `processed_data/babylm.pkl` or `processed_data/babylm_filtered.pkl` pickle files for the processed BabyLM data, follow the `pretrain_data_augment.ipynb` notebook to further augment the processed data via the `augment_and_save_chunks` function. The augmented chunks will be saved as `babylm_augment_{i}.pkl`.

5. Download the augmented chunks `babylm_augment_{i}.pkl` and put the files in `babylm_augment/babylm_augment_{i}.pkl`. Then, follow the first two cells which loads the augmented data and wiki data of BabyLM to build the final corpus. The corpus and the composition log will be saved as `babylm_pretrain_corpus/babylm_emo_wiki_{corpus_size}.pkl`, `babylm_pretrain_corpus/babylm_emo_wiki_{corpus_size}.txt`.

*In our project, we used the `babylm_filtered.pkl` file and augmented about 38 chunks of processed data to build the final corpus, the 10M corpus `babylm_pretrain_corpus/babylm_emo_wiki_10M.pkl` is used in subsequent experiments.*

---

