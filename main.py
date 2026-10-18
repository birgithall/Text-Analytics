import os
import json
import csv
import pandas as pd
from dotenv import load_dotenv
from huggingface_hub import hf_hub_download
import kagglehub
from kagglehub import KaggleDatasetAdapter

#Load environment variables from the .env file
load_dotenv()
hf_token = os.getenv("HF_TOKEN")
if not hf_token:
    raise ValueError("HF_TOKEN not found. Make sure you created a .env file with HF_TOKEN=hf_...")

#Load Hugging Face dataset
json_path = hf_hub_download(
    repo_id="allura-org/disco-elysium-conversations-raw",
    filename="output.json",
    repo_type="dataset",
    token=hf_token
)

with open(json_path, "r", encoding="utf-8") as f:
    hf_data = json.load(f)


#Download Kaggle dataset
kaggle_dir = kagglehub.dataset_download("lizakonopelko/disco-elysium-dialogue-texts")
csv_path = os.path.join(kaggle_dir, "texts_extracted.csv")

kaggle_df = pd.read_csv(
    csv_path,
    engine="python",
    on_bad_lines="skip",
    encoding="utf-8"
)

