import argparse
import os
import pickle

from tqdm import tqdm

from transformers import pipeline
from pretraining.data_pipeline import *
from pretraining.data_utils import *

parser = argparse.ArgumentParser(description='Pretraining data runner')
parser.add_argument('--dir_paths', nargs='+', default=None, help='Raw data directories paths. Default None (use Babylm and Tinystories data paths)')
parser.add_argument('--save_paths', nargs='+', default=None, help='Saving paths for processed data and results. Default None (use pre-defined paths and filenames)')
parser.add_argument('--sample_subset', type=int, default=0, help='Number of samples to use. Default 0 (use all data)')
parser.add_argument('--seed', type=int, default=42, help='Random seed for sampling data subsets, default 42')
parser.add_argument('--filter_emotions', type=bool, default=True, help='Filter non neutral emotions with scores > 0.5, it will save an extra file. Default True')
parser.add_argument('--model', type=str, default="SamLowe/roberta-base-go_emotions", help='Huggingface model to use for emotion classification, must be text-classification model. Default "SamLowe/roberta-base-go_emotions"')
parser.add_argument('--device', type=str, default="cpu", help='Device to use for model inference (cpu / cuda / mps). Default "cpu"')
args = parser.parse_args()

if __name__ == "__main__":
    # Set up the paths
    if args.dir_paths is None:
        dir_paths = ["data/babylm_data/babylm_100M/", "data/tinystories_data/"]
        print(f"Directory paths not provided, using default paths {dir_paths}")
    elif isinstance(args.dir_paths, str):
        dir_paths = [args.dir_paths]
    else:
        dir_paths = args.dir_paths

    if args.save_paths is None:
        save_paths = ["processed_data/babylm.pkl", "processed_data/tinystories.pkl"]
        print(f"Save paths not provided, using default paths {save_paths}")
    elif isinstance(args.save_paths, str):
        save_paths = [args.save_paths]
    else:
        save_paths = args.save_paths
    
    for save_path in save_paths:
        if not save_path.endswith(".pkl"):
            raise ValueError(f"Invalid save path {save_path}, must end with .pkl, the pipeline only supports pickle files.")
        
    # Fetch data and run the pipeline
    for i, (dir_path, save_path) in enumerate(zip(dir_paths, save_paths)):
        print(f"Fetching data files from {dir_path}, running {i+1}/{len(dir_paths)} paths")
        data_sources = [f for f in os.listdir(dir_path) if not f.startswith(".")]
        data_dict = get_data(dir_path, data_sources, args.sample_subset, args.seed)
        total = 0
        for source, ds in data_dict.items():
            print(f"{source} lines of corpus: {len(ds)}")
            total += len(ds)
        print(f"{dir_path} total lines of corpus: {total}")

        pipe = pipeline(
            "text-classification", 
            model=args.model, 
            top_k=None,
            framework="pt", # pytorch
            device=args.device # cpu / cuda / mps
            )
        
        results = classify_emotion(data_dict, pipe)

        print(f"Saving processed data to {save_path}")
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        pickle.dump(results, open(f"{save_path}", "wb"))

        if args.filter_emotions:
            print(f"Filtering emotions with scores > 0.5")
            results = [s for s in tqdm(results) if (s["label_0"] != "neutral") and (s["score_0"] > 0.5)]
            filtered_save_path = save_path.replace(".pkl", "_filtered.pkl")
            print(f"Saving filtered data to {filtered_save_path}")
            pickle.dump(results, open(f"{filtered_save_path}", "wb"))