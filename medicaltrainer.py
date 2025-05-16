"""
Trainer module for fine-tuning LLMs on medical conversation data.

This module handles the training loop and integration with HuggingFace's Trainer.
"""

class MedicalLLMTrainer:
    """
    Trainer for fine-tuning LLMs on medical conversation data.
    
    This class:
    1. Sets up the training environment
    2. Configures the model with PEFT/LoRA
    3. Handles the training loop with wandb logging
    4. Saves checkpoints and the final model
    """
    
    def __init__(
        self,
        model_config: ModelConfig,
        tokenizer: AutoTokenizer,
        peft_model: PeftAdapter,
        train_dataset: Dataset,
        val_dataset: Dataset,
        output_dir: Optional[str] = None,
        test_dataset: Optional[Dataset] = None
    ):
        """
        Initialize the trainer.
        
        Args:
            model_config: Configuration for the model and training
            output_dir: Directory to save outputs (uses config.output_dir if not provided)
            peft_adapter: Optional pre-configured PEFT adapter
        """
        self.config = model_config
        self.output_dir = output_dir or model_config.output_dir
        
        # Will be set during setup
        self.model = peft_model
        self.tokenizer = tokenizer
        self.trainer = None
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset
        self.sympoms = load_symptom_dataset()
        
        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)
    
        
    def _setup_wandb(self):
        """Set up Weights & Biases for experiment tracking."""
        if self.config.use_wandb:
            logger.info("Setting up wandb for experiment tracking")
            wandb.init(
                project=self.config.wandb_project,
                name=self.config.wandb_run_name,
                config=self.config.to_dict()
            )
    def train(
        self,
        training_args: Optional[TrainingArguments] = None,
        data_collator: Optional[Any] = None,
        resume_from_checkpoint: Optional[str] = None
    ) -> Tuple[Trainer, Dict[str, float]]:
        """
        Train the model on the provided dataset.
        
        Args:
            train_dataset: Training dataset
            eval_dataset: Optional evaluation dataset
            training_args: Optional custom training arguments
            data_collator: Optional custom data collator
            resume_from_checkpoint: Optional path to checkpoint to resume from
            
        Returns:
            Tuple of (trainer, training_results)
        """
        # Set up wandb if enabled
        if self.config.use_wandb:
            self._setup_wandb()
        
        # Set up training arguments
        if training_args is None:
            training_args = TrainingArguments(
                output_dir=self.output_dir,
                num_train_epochs=self.config.num_train_epochs,
                per_device_train_batch_size=self.config.per_device_train_batch_size,
                per_device_eval_batch_size=self.config.per_device_eval_batch_size,
                gradient_accumulation_steps=self.config.gradient_accumulation_steps,
                learning_rate=self.config.learning_rate,
                weight_decay=self.config.weight_decay,
                max_grad_norm=self.config.max_grad_norm,
                warmup_steps=self.config.warmup_steps,
                logging_steps=self.config.logging_steps,
                save_steps=self.config.save_steps,
                eval_steps=self.config.eval_steps,
                save_total_limit=self.config.save_total_limit,
                eval_strategy="steps" if self.val_dataset is not None else "no",
                lr_scheduler_type=self.config.lr_scheduler_type,
                optim=self.config.optimizer,
                fp16=self.config.fp16,
                gradient_checkpointing=self.config.gradient_checkpointing,
                report_to="wandb" if self.config.use_wandb else "none",
                seed=self.config.seed,
                push_to_hub=self.config.push_to_hub,
                hub_model_id=self.config.hub_model_id
            )
        
        # Set up data collator if not provided
        if data_collator is None:
            data_collator = transformers.DataCollatorForSeq2Seq(
                self.tokenizer, 
                pad_to_multiple_of=8,
                return_tensors="pt",
                padding=True
            )
        
        # Set up trainer
        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.val_dataset,
            tokenizer=self.tokenizer,
            data_collator=data_collator
        )
            # Convert to context-aware trainer if using context weighting
        if args.context_weight > 1.0:
            logger.info(f"Using context-aware training with weight {args.context_weight}")

            # First prepare symptom token IDs for context weighting
            # This is a simplified approach - in practice you would want to be more comprehensive
            common_symptoms = symptoms
            symptom_token_ids = []
            for symptom in common_symptoms:
                tokens = self.tokenizer.encode(symptom, add_special_tokens=False)
                symptom_token_ids.extend(tokens)

            # Remove duplicates
            symptom_token_ids = list(set(symptom_token_ids))

            logger.info(f"Using {len(symptom_token_ids)} symptom token IDs for context weighting")

            # Train the model
            trainer.train(
                train_dataset=self.train_dataset,
                eval_dataset=self.eval_dataset
            )

            # Convert to context-aware trainer after initial setup
            context_trainer = ContextAwareTrainer.from_standard_trainer(
                trainer=trainer.trainer,  # Access the underlying HF Trainer
                context_weight=args.context_weight,
                symptom_token_ids=symptom_token_ids,
                enable_context_weighting=True
            )

            # Continue training with context weighting
            context_trainer.train(resume_from_checkpoint=True)

            # Save the final model
            trainer.save_model()
        else:
            # Regular training without context weighting
            logger.info("Using standard training (no context weighting)")

            # Train the model
            trainer.train(
                train_dataset=train_dataset,
                eval_dataset=eval_dataset
            )
            # Train the model
            logger.info("Starting training...")
            train_result = self.trainer.train()
        
        # Save the final model
        self.save_model()
        
        # Log training results
        metrics = train_result.metrics
        self.trainer.log_metrics("train", metrics)
        self.trainer.save_metrics("train", metrics)
        
        # Run final evaluation if eval dataset is provided
        if self.val_dataset is not None:
            logger.info("Running final evaluation...")
            eval_metrics = self.trainer.evaluate()
            self.trainer.log_metrics("eval", eval_metrics)
            self.trainer.save_metrics("eval", eval_metrics)
            
            # Combine metrics
            metrics.update(eval_metrics)
        
        # Clean up wandb
        if self.config.use_wandb:
            wandb.finish()
            
        return self.trainer, metrics
    
    def save_model(self, output_dir: Optional[str] = None):
        """
        Save the trained model, tokenizer, and configuration.
        
        Args:
            output_dir: Directory to save to (uses self.output_dir if not provided)
        """
        output_dir = output_dir or os.path.join(self.output_dir, "final_model")
        os.makedirs(output_dir, exist_ok=True)
        
        logger.info(f"Saving model to {output_dir}")
        
        # Save through the PEFT adapter
        if self.peft_adapter:
            self.peft_adapter.save_model(output_dir)
        else:
            # Direct save if no adapter
            self.model.save_pretrained(output_dir)
            self.tokenizer.save_pretrained(output_dir)
            self.config.save(os.path.join(output_dir, "training_config.json"))
            
        logger.info(f"Model saved to {output_dir}")
        
        # Save to hub if configured
        if self.config.push_to_hub and self.config.hub_model_id:
            logger.info(f"Pushing model to hub: {self.config.hub_model_id}")
            self.model.push_to_hub(self.config.hub_model_id)
            self.tokenizer.push_to_hub(self.config.hub_model_id)
    
@classmethod
    def load_trained_model(
        cls,
        model_path: str,
        base_model_name: Optional[str] = None,
        device: str = "auto"
    ) -> Tuple[PeftModel, Any, ModelConfig]:
        """
        Load a trained model for inference.
        
        Args:
            model_path: Path to the saved model
            base_model_name: Optional base model name (loaded from config if not provided)
            device: Device to load the model on
            
        Returns:
            Tuple of (model, tokenizer, config)
        """
        # Load from PEFT adapter
        model, tokenizer, config = PeftAdapter.load_trained_model(
            model_path, 
            base_model_name
        )
        
        return model, tokenizer, config
    
    def evaluate_model(self):
    """Evaluate the trained model."""
        logger.info("Evaluating model...")
    
        # Initialize evaluator
        evaluator = MedicalConversationEvaluator(
            model=self.model,
            tokenizer=self.tokenizer,
            output_dir=os.path.join(self.config.output_dir, "evaluation")
        )
    
        # Evaluate all aspects of the model
        eval_results = evaluator.evaluate_all(
            with_markers=self.config.context_markers
        )
    
        # Log evaluation results
        logger.info("Evaluation results:")
        for aspect, results in eval_results.items():
            logger.info(f"{aspect}: {results}")
    
        # Save evaluation results
        results_path = os.path.join(args.output_dir, "evaluation_results.json")
        with open(results_path, "w") as f:
            json.dump(eval_results, f, indent=2)
    
        logger.info(f"Saved evaluation results to {results_path}")
    
        return eval_results
    
    def train_and_evaluate(
        self,
        **train_kwargs
    ) -> Dict[str, float]:
        """
        Convenience method to train and evaluate in one call.
        
        Args:
            train_dataset: Training dataset
            eval_dataset: Evaluation dataset
            **train_kwargs: Additional arguments to pass to train()
            
        Returns:
            Combined metrics from training and evaluation
        """
        trainer, metrics = self.train(
            **train_kwargs
        )
        eval_results = evaluate_model(
        
        return self.model, trainer, metrics, eval_results





"""
Context-aware training module for improving multi-turn conversation capabilities.

This module implements specialized training approaches to enhance contextual memory
in medical conversations, allowing the model to better track information across turns.
"""

import logging
import torch
import numpy as np
from typing import Dict, List, Optional, Union, Tuple, Any
from datasets import Dataset
from transformers import Trainer, TrainingArguments, PreTrainedModel, PreTrainedTokenizer
from loadsymptoms import load_symptom_dataset()
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

class ContextAwareTrainer(Trainer):
    """
    Extended Trainer with specialized capabilities for context-aware training.
    
    This trainer implements:
    1. Context-weighted loss to prioritize context retention
    3. Turn-based attention masking for improved conversational flow
    """
    
    def __init__(
        self,
        model: PreTrainedModel,
        args: TrainingArguments,
        train_dataset: Optional[Dataset] = None,
        eval_dataset: Optional[Union[Dataset, Dict[str, Dataset]]] = None,
        tokenizer: Optional[PreTrainedTokenizer] = None,
        data_collator: Optional[Any] = None,
        context_weight: float = 1.5,
        symptom_token_ids: Optional[List[int]] = None,
        enable_context_weighting: bool = True,
        **kwargs
    ):
        """
        Initialize the context-aware trainer.
        
        Args:
            model: The model to train
            args: Training arguments
            train_dataset: Training dataset
            eval_dataset: Evaluation dataset
            tokenizer: Tokenizer
            data_collator: Data collator
            context_weight: Weight to apply to context-related tokens in the loss
            symptom_token_ids: List of token IDs for symptom terms to prioritize
            enable_context_weighting: Whether to enable context-weighted loss
            **kwargs: Additional arguments to pass to the Trainer
        """
        super().__init__(
            model=model,
            args=args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            data_collator=data_collator,
            **kwargs
        )
        
        self.context_weight = context_weight
        self.symptom_token_ids = symptom_token_ids or []
        self.enable_context_weighting = enable_context_weighting
        self.symptoms = load_symptom_dataset()
        # If no symptom token IDs provided, try to generate from tokenizer if available
        if not self.symptom_token_ids and tokenizer is not None:
            self._generate_symptom_token_ids(tokenizer)
            
        logger.info(f"Initialized ContextAwareTrainer with context_weight={context_weight}")
        if self.symptom_token_ids:
            logger.info(f"Using {len(self.symptom_token_ids)} symptom token IDs for context weighting")
    
    def _generate_symptom_token_ids(self, tokenizer: PreTrainedTokenizer):
        """
        Generate token IDs for common symptom terms.
        
        Args:
            tokenizer: The tokenizer to use
        """
        common_symptoms = self.symptoms
        
        # Get token IDs for common symptoms
        for symptom in common_symptoms:
            tokens = tokenizer.encode(symptom, add_special_tokens=False)
            self.symptom_token_ids.extend(tokens)
            
        # Also add context marker tokens if they exist
        context_markers = ["[Previously mentioned:"]
        for marker in context_markers:
            tokens = tokenizer.encode(marker, add_special_tokens=False)
            self.symptom_token_ids.extend(tokens)
            
        # Remove duplicates
        self.symptom_token_ids = list(set(self.symptom_token_ids))
        
        logger.info(f"Generated {len(self.symptom_token_ids)} symptom token IDs")
    
    def compute_loss(self, model, inputs, return_outputs=False):
        """
        Compute training loss with context weighting.
        
        Args:
            model: The model
            inputs: Input batch
            return_outputs: Whether to return outputs along with the loss
            
        Returns:
            Loss value, or tuple of (loss, outputs) if return_outputs=True
        """
        if not self.enable_context_weighting:
            # Use standard loss calculation if context weighting is disabled
            return super().compute_loss(model, inputs, return_outputs)
        
        # Standard forward pass
        outputs = model(**inputs)
        loss = outputs.loss
        
        # Apply context weighting if enabled and we have symptom tokens
        if self.enable_context_weighting and self.symptom_token_ids and self.context_weight != 1.0:
            # Get logits and labels
            logits = outputs.logits
            labels = inputs["labels"]
            
            # Create a weight tensor (default weight is 1.0)
            weights = torch.ones_like(labels, dtype=torch.float)
            
            # Increase weight for symptom tokens
            for token_id in self.symptom_token_ids:
                weights = torch.where(labels == token_id, 
                                    torch.tensor(self.context_weight, device=weights.device),
                                    weights)
            
            # Also increase weight for tokens following context markers
            # This helps the model pay attention to the entire context section
            if "[Mentioned symptoms:" in self.symptom_token_ids:
                context_start_positions = (labels == self.symptom_token_ids[self.symptom_token_ids.index("[Mentioned symptoms:")])
                # Set higher weight for the next 20 tokens after each context marker
                for i in range(1, 20):
                    shifted = torch.roll(context_start_positions, shifts=i, dims=1)
                    shifted[:, :i] = False  # Clear wrapped values
                    weights = torch.where(shifted, torch.tensor(self.context_weight, device=weights.device), weights)
            
            # Compute weighted loss
            loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
            unweighted_loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))
            
            # Apply weights
            weighted_loss = unweighted_loss * weights.view(-1)
            
            # Replace the original loss
            loss = weighted_loss.mean()
        
        return (loss, outputs) if return_outputs else loss
    
    @classmethod
    def from_standard_trainer(
        cls,
        trainer: Trainer,
        context_weight: float = 1.5,
        symptom_token_ids: Optional[List[int]] = None,
        enable_context_weighting: bool = True
    ) -> 'ContextAwareTrainer':
        """
        Create a ContextAwareTrainer from a standard Trainer.
        
        Args:
            trainer: The standard Trainer to convert
            context_weight: Weight for context-related tokens
            symptom_token_ids: Token IDs for symptoms
            enable_context_weighting: Whether to enable context weighting
            
        Returns:
            ContextAwareTrainer instance
        """
        return cls(
            model=trainer.model,
            args=trainer.args,
            train_dataset=trainer.train_dataset,
            eval_dataset=trainer.eval_dataset,
            tokenizer=trainer.tokenizer,
            data_collator=trainer.data_collator,
            context_weight=context_weight,
            symptom_token_ids=symptom_token_ids,
            enable_context_weighting=enable_context_weighting,
            compute_metrics=trainer.compute_metrics,
            callbacks=trainer.callback_handler.callbacks,
            optimizers=trainer.optimizer_and_scheduler
        )

def create_context_aware_training_mix(
    standard_dataset: Dataset,
    context_dataset: Dataset,
    mix_ratio: float = 0.5
) -> Dataset:
    """
    Create a mixed dataset with standard and context-augmented examples.
    
    Args:
        standard_dataset: Dataset without context markers
        context_dataset: Dataset with context markers
        mix_ratio: Ratio of context examples (0.0 to 1.0)
        
    Returns:
        Mixed dataset
    """
    # Calculate how many examples to take from each dataset
    total_size = len(standard_dataset)
    context_size = int(total_size * mix_ratio)
    standard_size = total_size - context_size
    
    # Sample from each dataset
    standard_sample = standard_dataset.select(range(standard_size))
    context_sample = context_dataset.select(range(context_size))
    
    # Combine the datasets
    mixed_dataset = standard_sample.concatenate(context_sample)
    
    # Shuffle the dataset
    mixed_dataset = mixed_dataset.shuffle(seed=42)
    
    return mixed_dataset
