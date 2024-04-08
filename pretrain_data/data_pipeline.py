from tqdm import tqdm

from pretrain_data.data_utils import ListDataset, line_processor
# Parse data and preprocessing
# Feel free to add rules to filter lines

def get_data(dir_path, data_sources, sample_subset=0, seed=42):
    data = {}
    for source in data_sources:
        source_path = dir_path + source
        with open(source_path, "r") as f:
            processed = [l for l in map(line_processor, tqdm(f.readlines())) if l] # remove empty lines
        processed = list(set(processed)) # remove duplicates
        data[source_path] = ListDataset(processed, sample_subset, seed)
    return data

# Classify emotions of texts in dictionary format of {source: [line1, line2, ...]}
def classify_emotion(texts, pipe):
    results = []
    for source, ds in texts.items():
        print(f"Infering emotions for corpus in {source}")
        for i, scores in enumerate(tqdm(pipe(ds, truncation=True, padding=True))):
            # store the metadata and classification scores in a list of dictionaries
            res_dict = {"source": source, "text": ds[i]}
            # unnest the list of scores dictionary
            res_dict.update({f'{k}_{i}': v for i, d in enumerate(scores) for k, v in d.items()})
            results += [res_dict]
    return results