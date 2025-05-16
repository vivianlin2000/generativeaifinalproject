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
Model configuration for LLM fine-tuning.

This module contains the configuration settings for the Llama 2 model
and PEFT (Parameter-Efficient Fine-Tuning) setup.
"""

@dataclass
class ModelConfig:
    """Configuration settings for the LLM model and training process."""
    
    # Base model settings
    base_model_name: str = "meta-llama/Llama-2-7b"
    use_4bit: bool = True  # Use 4-bit quantization for memory efficiency
    use_nested_quant: bool = False  # Use nested quantization for further memory optimization
    bnb_4bit_compute_dtype: str = "float16"  # Compute dtype for 4-bit quantization
    bnb_4bit_quant_type: str = "nf4"  # Quantization type
    
    # LoRA configuration
    lora_r: int = 16  # LoRA attention dimension
    lora_alpha: int = 32  # Alpha parameter for LoRA scaling
    lora_dropout: float = 0.05  # Dropout probability for LoRA layers
    
    # Target modules to apply LoRA to (specific to Llama architecture)
    target_modules: List[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ])
    
    # Training parameters
    num_train_epochs: int = 3
    learning_rate: float = 2e-4
    warmup_steps: int = 100
    weight_decay: float = 0.01
    max_grad_norm: float = 0.3
    optimizer: str = "adamw_torch"
    lr_scheduler_type: str = "cosine"
    fp16: bool = True  # Use mixed precision training
    
    # Batch size and sequence length settings
    per_device_train_batch_size: int = 4
    per_device_eval_batch_size: int = 4
    gradient_accumulation_steps: int = 2
    max_seq_length: int = 2048
    
    # Output settings
    output_dir: str = f"./training-output-{str(int(time.time()))}"
    save_steps: int = 500
    eval_steps: int = 500
    logging_steps: int = 100
    save_total_limit: int = 3
    
    # Additional training options
    seed: int = 42
    push_to_hub: bool = False
    hub_model_id: Optional[str] = None
    gradient_checkpointing: bool = True
    
    # Wandb logging
    use_wandb: bool = False
    wandb_project: str = "medical-llm-finetuning"
    wandb_run_name: Optional[str] = None
    
    # Context-specific settings for medical conversations
    # These are special parameters for our approach to enhance context retention
    add_special_tokens: bool = True
    medical_context_markers: bool = True
    max_context_turns: int = 5  # Maximum number of conversation turns to consider for context
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Set run name if not provided
        if self.wandb_run_name is None:
            model_short_name = self.base_model_name.split("/")[-1]
            self.wandb_run_name = f"medical-ft-{model_short_name}-r{self.lora_r}-a{self.lora_alpha}"
            
        # Log configuration
        logger.info(f"Initialized model configuration for {self.base_model_name}")
        logger.info(f"LoRA config: r={self.lora_r}, alpha={self.lora_alpha}, dropout={self.lora_dropout}")
        logger.info(f"Training parameters: lr={self.learning_rate}, epochs={self.num_train_epochs}")

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {k: v for k, v in self.__dict__.items() if not k.startswith('_')}
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'ModelConfig':
        """Create configuration from dictionary."""
        return cls(**config_dict)
    
    def save(self, path: str) -> None:
        """Save configuration to JSON file."""
        import json
        
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
        
        logger.info(f"Saved model configuration to {path}")
    
    @classmethod
    def load(cls, path: str) -> 'ModelConfig':
        """Load configuration from JSON file."""
        import json
        
        with open(path, 'r') as f:
            config_dict = json.load(f)
        
        logger.info(f"Loaded model configuration from {path}")
        return cls.from_dict(config_dict)

# Default configurations for different model sizes
DEFAULT_CONFIGS = {
    # 7B model configuration
    "7b": ModelConfig(
        base_model_name="meta-llama/Llama-2-7b-hf",
        lora_r=16,
        lora_alpha=32,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=2
    ),
    
    # 13B model configuration (reduced batch size)
    "13b": ModelConfig(
        base_model_name="meta-llama/Llama-2-13b-hf",
        lora_r=16,
        lora_alpha=32,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4
    ),
    
    # Small 7B configuration for limited GPU resources
    "7b-small": ModelConfig(
        base_model_name="meta-llama/Llama-2-7b-hf",
        lora_r=8,
        lora_alpha=16,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4
    ),
    "opt-1.3b": ModelConfig(
        base_model_name="facebook/opt-1.3b",
        lora_r=16,
        lora_alpha=32,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=2
    ),
    "flan-small": ModelConfig(
        base_model_name="google/flan-t5-small",
        lora_r=16,
        lora_alpha=32,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=2
    ),
    "flan-base": ModelConfig(
        base_model_name="google/flan-t5-basae",
        lora_r=16,
        lora_alpha=32,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=2
    )
    
}

def get_config(model_size: str = "7b-small") -> ModelConfig:
    """Get default configuration for a specific model size."""
    if model_size not in DEFAULT_CONFIGS:
        logger.warning(f"Model size {model_size} not found in DEFAULT_CONFIGS, using '7b-small' instead")
        model_size = "7b-small"
        
    return DEFAULT_CONFIGS[model_size]
