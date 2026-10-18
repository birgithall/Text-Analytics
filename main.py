import os
import json
import csv
import pandas as pd
from dotenv import load_dotenv
from huggingface_hub import hf_hub_download
import kagglehub
from kagglehub import KaggleDatasetAdapter

#-----------Loading Datasets-----------

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


#-----------Skill Hierarchy-----------

SKILL_HIERARCHY = {
    "Intellect": [
        "Logic", "Encyclopedia", "Rhetoric", 
        "Drama", "Conceptualization", "Visual Calculus"
    ],
    "Psyche": [
        "Volition", "Inland Empire", "Empathy", 
        "Authority", "Esprit de Corps", "Suggestion"
    ],
    "Physique": [
        "Endurance", "Pain Threshold", "Physical Instrument", 
        "Electrochemistry", "Shivers", "Half Light"
    ],
    "Motorics": [
        "Hand/Eye Coordination", "Perception", "Reaction Speed", 
        "Savoir Faire", "Interfacing", "Composure"
    ]
}

#Because our skill hierarchy is attribute to skills
#For faster filtering, we will create a reverse mapping from skill to attribute
SKILL_TO_ATTR = {}
for attr, skills in SKILL_HIERARCHY.items():
    for skill in skills:
        SKILL_TO_ATTR[skill.upper()] = (skill, attr)




#Extracting  spoken lines by the 24 inner skills
records = []
for conv_idx, turns in enumerate(hf_data["conversations"]):
    #Safety check, if a conversation is in a dict, grab the list inside it
    if isinstance(turns, dict):
        turns = next((v for v in turns.values() if isinstance(v, list)), [])
    if not isinstance(turns, list):
        continue
    

    #in order to have conversation context we are saving single line spoken right before the current one
    prev_text = ""
    for turn_idx, turn in enumerate(turns):
        if not isinstance(turn, dict):
            continue
        speaker_raw = str(turn.get("actor", "")).strip().upper()
        text = str(turn.get("dialogue", "")).strip()

        # merging all different Perception skills like "PERCEPTION (SIGHT)" into one Perception
        if speaker_raw.startswith("PERCEPTION"):
            speaker_clean = "PERCEPTION"
        elif speaker_raw == "HAND/EYE COORDINATION":
            speaker_clean = "HAND/EYE COORDINATION"
        else:
            speaker_clean = speaker_raw

        #Saving only the lines spoken by the 24 inner skills    
        if speaker_clean in SKILL_TO_ATTR and text:
            canonical_skill, attribute = SKILL_TO_ATTR[speaker_clean]
            records.append({
                "conv_id": conv_idx,
                "turn_id": turn_idx,
                "text": text,
                "prev_text": prev_text,
                "skill": canonical_skill,
                "attribute": attribute
            })
        
        if text:
            prev_text = text


df_skills = pd.DataFrame(records)
print(f"Total inner-voice lines extracted: {len(df_skills)}")
print("\nLines per Attribute:\n", df_skills["attribute"].value_counts())
print("\nLines per Skill:\n", df_skills["skill"].value_counts())