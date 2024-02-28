import re

import numpy as np

from torch.utils.data import Dataset

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

def line_processor(line):
    line = re.sub("[\t\n]", "", line) # remove tabs and newlines
    line = re.sub(r'\s+([.,!?;:])', r'\1', line) # remove spaces before punctuation
    line = line.strip() # remove leading and trailing spaces
    if len(line.split()) <= 10: # remove lines with less than 10 words
        return None
    return line