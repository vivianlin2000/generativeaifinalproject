import os
import logging
from typing import Dict, List, Optional, Union, Tuple, Any
from datasets import load_dataset, load_from_disk, Dataset, DatasetDict
import pandas as pd
import json
import re
import numpy as np
import subprocess
import tempfile
import shutil
from dataclasses import dataclass, field
import time
import gc
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, Trainer, TrainingArguments
import wandb
import torch

import transformers
from datasets import Dataset


from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    LlamaForCausalLM,
    LlamaTokenizer,
    Trainer, 
    TrainingArguments,
    DataCollatorForLanguageModeling,
    EarlyStoppingCallback
)
from peft import (
    LoraConfig,
    TaskType,
    get_peft_model,
    PeftModel,
    prepare_model_for_kbit_training
)

from google.colab import drive
drive.mount('/content/drive')

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

def load_symptoms() -> Dataset:
    """Load the dhivyeshrk/Disease-Symptom-Extensive-Clean dataset from HuggingFace."""
    logger.info("Loading dhivyeshrk/Disease-Symptom-Extensive-Clean dataset...")
    try:
        dataset = load_dataset("dhivyeshrk/Disease-Symptom-Extensive-Clean")
        logger.info(f"dhivyeshrk/Disease-Symptom-Extensive-Clean dataset loaded. Size: {len(dataset['train'])}")
        return dataset
    except Exception as e:
        logger.error(f"Error loading dhivyeshrk/Disease-Symptom-Extensive-Clean dataset: {e}")
        raise

def load_symptom_dataset() -> Dict[str, Dataset]:
    """Load all available datasets."""
    try:
        symptom_dataset = load_symptoms()['train']
        symptom_df = symptom_dataset.to_pandas()
        symptoms = list(symptom_df.columns[1:])
        del(symptom_df)
        gc.collect()

        common_symptoms = [
        "pain", "ache", "fever", "cough", "headache", "nausea", "vomiting",
        "dizziness", "fatigue", "rash", "swelling", "inflammation", "bleeding",
        "infection", "shortness of breath", "weakness", "numbness", "diarrhea",
        "chest pain", "back pain", "sore throat", "runny nose", "congestion", "throbbing"]
        not_found_symptoms = [symptom for symptom in common_symptoms if symptom not in symptoms]
        symptoms.extend(not_found_symptoms)
        del(common_symptoms)
        del(not_found_symptoms)
        gc.collect()
        return symptoms
    except Exception as e:
        logger.warning(f"Could not load dhivyeshrk/Disease-Symptom-Extensive-Clean: {e}")
