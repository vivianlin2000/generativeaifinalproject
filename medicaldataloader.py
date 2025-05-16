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

"""
Data loader module for medical conversation datasets.
Handles downloading and loading datasets from HuggingFace and other sources.
"""
class MedicalDataLoader:
    """
    Loads and combines multiple medical conversation datasets.
    """

    def __init__(self, cache_dir: Optional[str] = "./data/cache"):
        """
        Initialize the data loader.

        Args:
            cache_dir: Directory to cache downloaded datasets
        """
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)

    def load_meddialog(self) -> Dataset:
        """Load the MedDialog dataset from HuggingFace."""
        logger.info("Loading MedDialog dataset...")
        try:
            dataset = load_dataset("bigbio/meddialog", cache_dir=self.cache_dir)
            logger.info(f"MedDialog dataset loaded. Size: {len(dataset['train'])}")
            return dataset
        except Exception as e:
            logger.error(f"Error loading MedDialog dataset: {e}")
            raise

    def load_healthcaremagic(self) -> Dataset:
        """Load the HealthCareMagic-100k dataset from HuggingFace."""
        logger.info("Loading HealthCareMagic dataset...")
        try:
            dataset = load_dataset("ZhexiLu/healthcaremagic-100k", cache_dir=self.cache_dir)
            logger.info(f"HealthCareMagic dataset loaded. Size: {len(dataset['train'])}")
            return dataset
        except Exception as e:
            logger.error(f"Error loading HealthCareMagic dataset: {e}")
            raise

    def load_icliniq(self) -> Dataset:
        """Load the iCliniq-10k dataset from HuggingFace."""
        logger.info("Loading iCliniq dataset...")
        try:
            dataset = load_dataset("wangrongsheng/icliniq-10k-en", cache_dir=self.cache_dir)
            logger.info(f"iCliniq dataset loaded. Size: {len(dataset['train'])}")
            return dataset
        except Exception as e:
            logger.error(f"Error loading iCliniq dataset: {e}")
            raise

    def load_meddialog1M(self) -> Dataset:
        """Load the MedDialog-1M dataset from HuggingFace."""
        logger.info("Loading MedDialog-1M dataset...")
        try:
            dataset = load_dataset("siyah1/med-dialogue-1M")
            logger.info(f"MedDialog-1M dataset loaded. Size: {len(dataset['train'])}")
            return dataset
        except Exception as e:
            logger.error(f"Error loading MedDialog-1M dataset: {e}")
            raise

    def load_all_datasets(self) -> Dict[str, Dataset]:
        """Load all available datasets."""
        datasets = {}

        try:
            datasets["meddialog"] = self.load_meddialog()
        except Exception as e:
            logger.warning(f"Could not load MedDialog: {e}")

        try:
            datasets["healthcaremagic"] = self.load_healthcaremagic()
        except Exception as e:
            logger.warning(f"Could not load HealthCareMagic: {e}")

        try:
            datasets["icliniq"] = self.load_icliniq()
        except Exception as e:
            logger.warning(f"Could not load iCliniq: {e}")

        try:
            datasets["meddialog-1M"] = self.load_meddialog1M()
        except Exception as e:
            logger.warning(f"Could not load MedDialog-1M: {e}")

        if not datasets:
            raise ValueError("Failed to load any datasets")

        return datasets

    def create_combined_dataset(self,
                               include_datasets: Optional[List[str]] = None,
                               train_split: float = 0.8,
                               validation_split: float = 0.1,
                               test_split: float = 0.1,
                               seed: int = 42) -> DatasetDict:
        """
        Create a combined dataset from multiple sources with train/val/test splits.

        Args:
            include_datasets: List of dataset names to include. If None, includes all available.
            train_split: Proportion to use for training (default: 0.8)
            validation_split: Proportion to use for validation (default: 0.1)
            test_split: Proportion to use for testing (default: 0.1)
            seed: Random seed for reproducible splits

        Returns:
            DatasetDict with train, validation, and test splits
        """
        if abs(train_split + validation_split + test_split - 1.0) > 1e-6:
            raise ValueError("Split proportions must sum to 1.0")

        available_datasets = self.load_all_datasets()

        if include_datasets is not None:
            datasets_to_use = {k: available_datasets[k] for k in include_datasets if k in available_datasets}
            if not datasets_to_use:
                raise ValueError(f"None of the specified datasets {include_datasets} are available")
        else:
            datasets_to_use = available_datasets

        logger.info(f"Creating combined dataset from: {list(datasets_to_use.keys())}")

        # First, standardize the dataset formats to ensure they can be combined
        standardized_datasets = []

        for dataset_name, dataset in datasets_to_use.items():
            if "train" in dataset:
                standardized = self._standardize_dataset_format(dataset["train"], dataset_name)
                standardized_datasets.append(standardized)
            else:
                standardized = self._standardize_dataset_format(dataset, dataset_name)
                standardized_datasets.append(standardized)

        # Combine all standardized datasets
        combined_dataset = Dataset.from_pandas(
            pd.concat([ds.to_pandas() for ds in standardized_datasets])
        )

        # Create splits
        splits = combined_dataset.train_test_split(
            test_size=validation_split + test_split,
            seed=seed
        )
        train_dataset = splits["train"]

        # Further split the test portion into validation and test
        if test_split > 0:
            test_valid_split = splits["test"].train_test_split(
                test_size=test_split / (validation_split + test_split),
                seed=seed
            )
            valid_dataset = test_valid_split["train"]
            test_dataset = test_valid_split["test"]
        else:
            valid_dataset = splits["test"]
            test_dataset = Dataset.from_dict({"dialogue": [], "turns": [], "source": []})

        return DatasetDict({
            "train": train_dataset,
            "validation": valid_dataset,
            "test": test_dataset
        })

    def _standardize_dataset_format(self, dataset: Dataset, source_name: str) -> Dataset:
        """
        Standardize dataset to a common format for medical conversations.

        The standard format will have the following columns:
        - dialogue: The full conversation text
        - turns: List of dictionaries, each with 'role' and 'content'
        - source: The original dataset name

        Args:
            dataset: The dataset to standardize
            source_name: Name of the source dataset for tracking

        Returns:
            Standardized dataset
        """

        if source_name == "meddialog":
            # MedDialog format conversion
            df = dataset.to_pandas()

            # Create standardized dataframe
            result = []
            for _, row in df.iterrows():
                # Extract turns from the dialogue
                dialogue_text = ""
                turns = []
                for r in row["utterances"]["speaker"]:
                    # Check if it's a doctor or patient based on the speaker field
                    role = "doctor" if r % 2 == 1 else "patient"
                    turns.append({
                        "role": role,
                        "content": row["utterances"]["utterance"][r]
                    })
                    dialogue_text += f"{role}: {row['utterances']['utterance'][r]}\n"

                result.append({
                    "dialogue": dialogue_text,
                    "turns": turns,
                    "source": source_name
                })
            return Dataset.from_pandas(pd.DataFrame(result))

        elif source_name == "healthcaremagic":
            # HealthCareMagic format conversion
            df = dataset.to_pandas()

            result = []
            for _, row in df.iterrows():
                # Extract the question and answer
                question = row["input"]
                answer = row["output"]

                # Create a simple 2-turn dialogue
                dialogue_text = f"patient: {question}\ndoctor: {answer}\n"
                turns = [
                    {"role": "patient", "content": question},
                    {"role": "doctor", "content": answer}
                ]

                result.append({
                    "dialogue": dialogue_text,
                    "turns": turns,
                    "source": source_name
                })

            return Dataset.from_pandas(pd.DataFrame(result))

        elif source_name == "icliniq":
            # iCliniq format conversion
            df = dataset.to_pandas()

            result = []
            for _, row in df.iterrows():
                # Extract the question and answer
                question = row["input"]
                answer = row["answer_icliniq"]

                # Create a simple 2-turn dialogue
                dialogue_text = f"patient: {question}\ndoctor: {answer}\n"
                turns = [
                    {"role": "patient", "content": question},
                    {"role": "doctor", "content": answer}
                ]

                result.append({
                    "dialogue": dialogue_text,
                    "turns": turns,
                    "source": source_name
                })

            return Dataset.from_pandas(pd.DataFrame(result))

        elif source_name == "meddialog-1M":
            # iCliniq format conversion
            df = dataset.to_pandas()
            result = []
            for id in set(df["conversation_id"]):
                dialogue_text = ""
                turns = []
                convo_df = df[df["conversation_id"] == id]
                for _, row in convo_df.iterrows():
                    # Extract the question and answer
                    # Check if it's a doctor or patient based on the speaker field
                    turns.append({
                        "role": row["sender"],
                        "content": row["content"]
                    })
                    dialogue_text += f"{row['sender']}: {row['content']}\n"
                result.append({
                    "dialogue": dialogue_text,
                    "turns": turns,
                    "source": source_name
                })

            return Dataset.from_pandas(pd.DataFrame(result))

        else:
            raise ValueError(f"Unknown dataset format: {source_name}")
