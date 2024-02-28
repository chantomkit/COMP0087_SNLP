import re
import pickle

import pandas as pd
import numpy as np

from torch.utils.data import Dataset
from transformers import pipeline

from tqdm import tqdm


class ListDataset(Dataset):
    def __init__(self, original_list, sample_subset=0, seed=42):
        if sample_subset and sample_subset < len(original_list):
            np.random.seed(seed)
            np.random.shuffle(original_list)
            self.ds_list = original_list[:sample_subset]
        else:
            self.ds_list = original_list
    def __len__(self):
        return len(self.ds_list)

    def __getitem__(self, i):
        return self.ds_list[i]

# Parse data and preprocessing
# Feel free to add rules to filter lines
def line_processor(line):
    line = re.sub("[\t\n]", "", line) # remove tabs and newlines
    line = re.sub(r'\s+([.,!?;:])', r'\1', line) # remove spaces before punctuation
    line = line.strip() # remove leading and trailing spaces
    if len(line.split()) <= 10: # remove lines with less than 10 words
        return None
    return line

def get_data(data_sources, sample_subset=0, seed=42):
    data = {}
    for source in data_sources:
        source_path = dir_path + source
        with open(source_path, "r") as f:
            processed = [l for l in map(line_processor, f.readlines()) if l] # remove empty lines
            processed = list(set(processed)) # remove duplicates
            data[source_path] = ListDataset(processed, sample_subset, seed)
    return data

# Classify emotions of texts in dictionary format of {source: [line1, line2, ...]}
def classify_emotion(texts, pipe):
    results = []
    for source, ds in texts.items():
        print(f"Processing {source}")
        for i, scores in enumerate(tqdm(pipe(ds, truncation=True, padding=True))):
            # store the metadata and classification scores in a list of dictionaries
            res_dict = {"source": source, "text": ds[i]}
            # unnest the list of scores dictionary
            res_dict.update({f'{k}_{i}': v for i, d in enumerate(scores) for k, v in d.items()})
            results += [res_dict]
    return results


def filter_emotion(df, threshold=0.5):
    filtered_df = df.copy()
    scores_df = df.filter(regex="^score_")
    labels_df = df.filter(regex="^label_")
    # retrieve confident label(s) with scores >= threshold
    labels = labels_df.where((scores_df >= threshold).values).values
    # remove nans
    labels = [tuple(set(l[~pd.isnull(l)]))for l in labels]
    filtered_df["filtered_labels"] = labels
    filtered_df["num_labels"] = [len(l) for l in labels]
    return filtered_df



if __name__ == '__main__':
    dir_path = f"data/babylm_data/babylm_100M/"
    data_sources = [
            "aochildes.train", 
            "bnc_spoken.train", 
            "cbt.train",
            "children_stories.train",
            "gutenberg.train",
            "open_subtitles.train",
            "qed.train",
            "simple_wikipedia.train",
            "switchboard.train",
            "wikipedia.train"
        ]
    
    texts = get_data(data_sources, sample_subset=0)

    # Select emotion classification model

    # Faster model, less emotions category with worse accuracy
    # pipe = pipeline(
    #     "text-classification", 
    #     model="j-hartmann/emotion-english-distilroberta-base", 
    #     top_k=None,
    #     framework="pt", # pytorch
    #     )

    # Larger model, more emotions category with better accuracy, slower inference
    # NOTE: This model oftens predict neutral emotion
    pipe = pipeline(
        "text-classification", 
        model="SamLowe/roberta-base-go_emotions", 
        top_k=None,
        framework="pt", # pytorch
        device="cuda" # cpu / cuda / mps
        )

    scores = classify_emotion(texts, pipe)

    pickle.dump(scores, open("processed_data/babylm.pkl", "wb"))

    # futher filter out neutral emotions
    scores = pickle.load(open("processed_data/babylm.pkl", "rb"))
    df = pd.DataFrame(scores)
    set(df.iloc[0].filter(regex="^label_").values) # all possible emotions
    filtered_df = filter_emotion(df)

    emotion_freq = filtered_df.filtered_labels.value_counts()
    emotion_freq.head(40) # label statistics

    # drop if neutral or no emotion
    tmp_df = filtered_df[~filtered_df.filtered_labels.apply(lambda x: "neutral" in x or not x)]
    tmp_df.groupby("source").filtered_labels.value_counts().groupby(level=0).head(5)
    tmp_df.to_csv("processed_data/babylm_filtered.csv", index=False)