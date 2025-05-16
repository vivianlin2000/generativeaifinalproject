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
Data processor for preparing medical conversation data for LLM fine-tuning.
"""
class MedicalDataProcessor:
    """
    Processes medical conversation data for LLM fine-tuning.

    This class handles:
    1. Cleaning and normalizing conversations
    2. Converting to instruction format
    3. Adding context markers for multi-turn conversations
    4. Filtering conversations based on quality criteria
    """

    def __init__(
        self,
        max_sequence_length: int = 512,
        min_turns: int = 2,
        include_context_markers: bool = True,
        system_prompt: Optional[str] = None
    ):
        """
        Initialize the data processor.

        Args:
            max_sequence_length: Maximum sequence length for model inputs
            min_turns: Minimum number of conversation turns to include
            include_context_markers: Whether to add special context markers
            system_prompt: Optional system prompt to prepend to conversations
        """
        self.max_sequence_length = max_sequence_length
        self.min_turns = min_turns
        self.include_context_markers = include_context_markers

        # Default system prompt for medical conversations if none provided
        self.system_prompt = system_prompt or """You are a helpful AI assistant providing information on health-related topics. Maintain awareness of the conversation history and any symptoms or conditions mentioned previously. Never provide a diagnosis, but offer evidence-based information and suggest consulting healthcare professionals when appropriate. Always prioritize patient safety above all else."""

    def clean_text(self, text: str) -> str:
        """Clean and normalize text content."""
        if not text or not isinstance(text, str):
            return ""

        # Remove multiple spaces
        text = re.sub(r'\s+', ' ', text)

        # Remove HTML tags
        text = re.sub(r'<[^>]+>', '', text)

        # Normalize quotes and apostrophes
        text = text.replace(''', "'").replace(''', "'").replace('"', '"').replace('"', '"')

        # Fix common medical abbreviations spacing
        text = re.sub(r'([a-zA-Z])(\d+)', r'\1 \2', text)  # Add space between letters and numbers

        return text.strip()

    def filter_conversation(self, conversation: Dict[str, Any]) -> bool:
        """
        Filter conversations based on quality criteria.

        Args:
            conversation: Dictionary with conversation data

        Returns:
            True if conversation should be kept, False if it should be filtered out
        """
        turns = conversation.get("turns", [])

        # Filter by minimum number of turns
        if len(turns) < self.min_turns:
            return False

        # Check for empty content
        if any(not turn.get("content", "").strip() for turn in turns):
            return False

        # Check for minimum content length in patient turns
        patient_turns = [t for t in turns if t.get("role") == "patient"]
        if patient_turns and max(len(t.get("content", "")) for t in patient_turns) < 10:
            return False

        # Check for minimum content length in doctor turns
        doctor_turns = [t for t in turns if t.get("role") == "doctor"]
        if doctor_turns and max(len(t.get("content", "")) for t in doctor_turns) < 20:
            return False

        return True

    def add_context_markers(self, turns: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """
        Add context markers to conversation turns to enhance context awareness.

        This adds special tokens or markers that highlight previously mentioned
        symptoms or concerns, to encourage the model to maintain awareness of
        the full conversation history.

        Args:
            turns: List of conversation turns with 'role' and 'content'

        Returns:
            Updated turns with context markers
        """
        if not self.include_context_markers:
            return turns

        # Simple medical term extraction function (to be expanded with proper NER in practice)
        def extract_medical_terms(text):
            # This is a simplified version - in a real implementation,
            # you would use medical NER models or term dictionaries
            common_symptoms = symptoms
            text_lower = text.lower()
            found_terms = [term for term in common_symptoms if term in text_lower]
            return found_terms

        # Copy the turns to avoid modifying the original
        marked_turns = []
        all_mentioned_terms = set()

        for i, turn in enumerate(turns):
            turn_copy = turn.copy()
            content = self.clean_text(turn_copy["content"])

            # Extract potential medical terms from this turn
            if turn["role"] == "patient":
                new_terms = extract_medical_terms(content)
                all_mentioned_terms.update(new_terms)

            # For doctor responses (not the first turn), add context markers
            if turn["role"] == "patient":
                context_prefix = "[Mentioned symptoms: " + ", ".join(all_mentioned_terms) + "]"
                # context_prefix = "[Previously mentioned: " + ", ".join(all_mentioned_terms) + "]"
                turn_copy["content"] = content + context_prefix

            marked_turns.append(turn_copy)
        return marked_turns

    def convert_to_instruction_format(
        self,
        conversation: Dict[str, Any],
        include_system_prompt: bool = True
    ) -> Dict[str, str]:
        """
        Convert conversation to instruction format for fine-tuning.

        Args:
            conversation: Dictionary with conversation data
            include_system_prompt: Whether to include the system prompt

        Returns:
            Dictionary with 'instruction' and 'response' fields
        """
        id = int(time.time())
        turns = conversation.get("turns", [])
        contexted_turns = None
        if not turns:
            return None

        # Apply context markers if enabled
        if self.include_context_markers:
            contexted_turns = self.add_context_markers(turns)
        else:
            contexted_turns = turns

        # For multi-turn conversations, we'll create multiple training examples
        # Each example will include all previous context
        training_examples = []

        for i in range(1, len(contexted_turns), 2):
            # Need at least a patient question and doctor answer
            if i >= len(contexted_turns):
                break

            # Get all turns up to this patient question
            context_turns = contexted_turns[:i]

            # The current doctor's answer is what we want to predict
            if i < len(contexted_turns):
                target_turn = turns[i]
            else:
                continue

            # Build the instruction from all previous turns
            instruction_parts = []

            # Add system prompt if requested
            if include_system_prompt and self.include_context_markers:
                instruction_parts.append(f"<s>[INST] <<SYS>>\n{self.system_prompt}\n<</SYS>>\n\n")
                # instruction_parts.append(f"{self.system_prompt}\n\n")
            elif include_system_prompt:
                instruction_parts.append(f"{self.system_prompt}\n\n")

            # Add previous context turns
            for turn in context_turns[:-1]:
                role_name = "Patient" if turn["role"] == "patient" else "Doctor"
                content = turn["content"]
                if role_name == "Patient":
                    content = f"{content} [/INST] "
                else:
                    content = f"{content} </s><s> [INST] "
                instruction_parts.append(f"{role_name}: {content}")
            # Add the current patient question
            current_patient_turn = contexted_turns[i-1]
            patient_question = current_patient_turn["content"]
            instruction_parts.append(f"Patient: {patient_question} [/INST]")

            # The response is the doctor's answer
            response = f"{self.clean_text(target_turn['content'])}"

            # Combine all parts
            instruction = "".join(instruction_parts)

            # Check if the combined length is within limit
            training_examples.append({
                    "instruction": instruction,
                    "response": response,
                    "conversation_id": conversation.get("id", f"conv_{id}"),
                    "turn_index": i
            })


        return training_examples

    def process_dataset(self, dataset: Dataset) -> Dataset:
        """
        Process the entire dataset for fine-tuning.

        Args:
            dataset: Dataset with conversation data

        Returns:
            Processed dataset ready for fine-tuning
        """
        logger.info(f"Processing dataset with {len(dataset)} conversations")

        # Filter conversations
        filtered_dataset = dataset.filter(
            self.filter_conversation,
            desc="Filtering conversations"
        )

        logger.info(f"Filtered to {len(filtered_dataset)} quality conversations")

        # Convert to instruction format
        def process_to_examples(example):
            examples = self.convert_to_instruction_format(example)
            if not examples:
                return {"instruction": [], "response": [], "conversation_id": [], "turn_index": []}

            instructions = [ex["instruction"] for ex in examples]
            responses = [ex["response"] for ex in examples]
            conv_ids = [ex["conversation_id"] for ex in examples]
            turn_indices = [ex["turn_index"] for ex in examples]

            return {
                "instruction": instructions,
                "response": responses,
                "conversation_id": conv_ids,
                "turn_index": turn_indices
            }

        processed_dataset = filtered_dataset.map(
            process_to_examples,
            remove_columns=dataset.column_names,
            batched=False,
            desc="Converting to instruction format"
        )

        # Explode the dataset (convert lists to individual examples)
        df = processed_dataset.to_pandas()
        exploded_df = pd.DataFrame({
            col: df[col].explode()
            for col in ["instruction", "response", "conversation_id", "turn_index"]
        }).reset_index(drop=True)

        # Remove rows with empty instructions or responses
        exploded_df = exploded_df.dropna(subset=["instruction", "response"])

        # Create the final dataset
        final_dataset = Dataset.from_pandas(exploded_df)

        logger.info(f"Created {len(final_dataset)} training examples from conversations")
        del(df)
        del(exploded_df)
        gc.collect()
        return final_dataset
