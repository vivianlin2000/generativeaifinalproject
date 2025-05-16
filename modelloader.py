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
PEFT adapter for Parameter-Efficient Fine-Tuning of LLMs.

This module implements LoRA (Low-Rank Adaptation) for fine-tuning
large language models efficiently on limited GPU resources.
"""


class PeftAdapter:
    """
    Handles setting up PEFT (Parameter-Efficient Fine-Tuning) for LLMs.
    
    This class supports:
    1. Loading and configuring base models with quantization
    2. Setting up LoRA adapters
    3. Saving and loading PEFT checkpoints
    """
    
    def __init__(self, config: ModelConfig):
        """
        Initialize the PEFT adapter.
        
        Args:
            config: Model and training configuration
        """
        self.config = config
        self.tokenizer = None
        self.model = None
        self.peft_config = None
    
    def load_base_model(self) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
        """
        Load the base model and tokenizer with quantization settings.
        
        Returns:
            Tuple of (model, tokenizer)
        """
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Loading base model: {self.config.base_model_name}")
        
        # Set up quantization configuration for memory-efficient loading
        compute_dtype = getattr(torch, self.config.bnb_4bit_compute_dtype)
        
        bnb_config = None
        if self.config.use_4bit:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type=self.config.bnb_4bit_quant_type,
                bnb_4bit_compute_dtype=compute_dtype,
                bnb_4bit_use_double_quant=self.config.use_nested_quant,
            )
            logger.info("Using 4-bit quantization for model loading")
        
        # Load the model with quantization settings
        model = AutoModelForCausalLM.from_pretrained(
            self.config.base_model_name,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
        ).to(device)
        
        # Prepare model for k-bit training if using quantization
        if self.config.use_4bit:
            model = prepare_model_for_kbit_training(model)
        
        # Enable gradient checkpointing for memory efficiency
        if self.config.gradient_checkpointing:
            model.gradient_checkpointing_enable()
            logger.info("Gradient checkpointing enabled")
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            self.config.base_model_name,
            trust_remote_code=True,
            use_fast=True,  # Use fast tokenizer if available
        )
        if self.config.base_model_name in ["meta-llama/Llama-2-7b-hf","meta-llama/Llama-2-13b-hf"]:
            tokenizer.add_special_tokens({'pad_token': '[PAD]'})
            
        # Resize token embeddings if special tokens were added
        model.resize_token_embeddings(len(tokenizer))
        
        logger.info(f"Model loaded: {self.config.base_model_name}")
        logger.info(f"Model has {sum(p.numel() for p in model.parameters())/1e9:.2f} billion parameters")
        
        self.model = model
        self.tokenizer = tokenizer
        
        return model, tokenizer
    
    def setup_peft(self, model: Optional[AutoModelForCausalLM] = None) -> PeftModel:
        """
        Set up PEFT (LoRA) on the model.
        
        Args:
            model: Model to adapt with PEFT (uses self.model if not provided)
            
        Returns:
            PEFT adapted model
        """
        model = model or self.model
        if model is None:
            raise ValueError("No model provided or loaded. Call load_base_model first.")
        
        logger.info(f"Setting up LoRA with r={self.config.lora_r}, alpha={self.config.lora_alpha}")
        
        # Configure LoRA
        self.peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=self.config.lora_r,
            lora_alpha=self.config.lora_alpha,
            lora_dropout=self.config.lora_dropout,
            target_modules=self.config.target_modules,
            bias="none",
            inference_mode=False,
        )
        
        # Apply LoRA to the model
        peft_model = get_peft_model(model, self.peft_config)
        
        # Log trainable parameters
        logger.info(f"PEFT model set up with {peft_model.print_trainable_parameters()}")
        
        self.model = peft_model
        return peft_model
    
    def get_model_and_tokenizer(self) -> Tuple[PeftModel, AutoTokenizer]:
        """
        Get the current model and tokenizer.
        
        Returns:
            Tuple of (peft_model, tokenizer)
        """
        if self.model is None or self.tokenizer is None:
            raise ValueError("Model and tokenizer not loaded. Call load_base_model first.")
            
        return self.model, self.tokenizer
    
    def save_model(self, output_dir: Optional[str] = None) -> str:
        """
        Save the PEFT model and tokenizer.
        
        Args:
            output_dir: Directory to save to (uses config.output_dir if not provided)
            
        Returns:
            Path where model was saved
        """
        output_dir = output_dir or self.config.output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        logger.info(f"Saving PEFT model to {output_dir}")
        
        # Save PEFT adapter weights (much smaller than full model)
        self.model.save_pretrained(output_dir)
        
        # Save tokenizer
        self.tokenizer.save_pretrained(output_dir)
        
        # Save configuration
        config_path = os.path.join(output_dir, "training_config.json")
        self.config.save(config_path)
        
        logger.info(f"Model, tokenizer, and configuration saved to {output_dir}")
        return output_dir
    
    @classmethod
    def load_trained_model(
        cls, 
        adapter_path: str, 
        base_model_name: Optional[str] = None
    ) -> Tuple[PeftModel, AutoTokenizer, ModelConfig]:
        """
        Load a trained PEFT model, tokenizer, and configuration.
        
        Args:
            adapter_path: Path to the saved PEFT adapter
            base_model_name: Base model name (loaded from config if not provided)
            
        Returns:
            Tuple of (peft_model, tokenizer, config)
        """
        # Load configuration
        config_path = os.path.join(adapter_path, "training_config.json")
        if os.path.exists(config_path):
            config = ModelConfig.load(config_path)
            logger.info(f"Loaded configuration from {config_path}")
        else:
            logger.warning(f"No configuration found at {config_path}, using default")
            config = ModelConfig()
        
        # Override base model name if provided
        if base_model_name:
            config.base_model_name = base_model_name
            
        # Initialize adapter with loaded config
        adapter = cls(config)
        
        # Load base model and tokenizer
        model, tokenizer = adapter.load_base_model()
        
        # Load the trained PEFT adapter
        logger.info(f"Loading trained PEFT adapter from {adapter_path}")
        model = PeftModel.from_pretrained(model, adapter_path)
        
        adapter.model = model
        adapter.tokenizer = tokenizer
        
        return model, tokenizer, config

def create_peft_model(config: ModelConfig) -> Tuple[PeftModel, AutoTokenizer]:
    """
    Helper function to create a PEFT model from configuration.
    
    Args:
        config: Model and training configuration
        
    Returns:
        Tuple of (peft_model, tokenizer)
    """
    adapter = PeftAdapter(config)
    base_model, tokenizer = adapter.load_base_model()
    peft_model = adapter.setup_peft()
    return base_model, tokenizer, peft_model
