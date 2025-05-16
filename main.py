from medicaldataloader import MedicalDataLoader
from loadsymptoms import load_symptom_dataset
from medicaldataprocessor import MedicalDataProcessor
from modelconfigs import get_config
from medicaltrainer import MedicalLLMTrainer
from modelloader import create_peft_model

def tokenize_function(example):
    labels = tokenizer(example['response'], padding="max_length", max_length=4096, truncation=True, return_tensors="pt").to(device)

    model_inputs = tokenizer(example['instruction'], padding="max_length", max_length=4096, truncation=True, return_tensors="pt").to(device)
    model_inputs["labels"] = labels["input_ids"]

    del(labels)
    gc.collect()
    return model_inputs
      
loader = MedicalDataLoader()
combined = loader.create_combined_dataset()
symptoms = load_symptom_dataset()
processor = MedicalDataProcessor(include_context_markers=True)
config = get_config(7b")
config.num_train_epochs = 1  
base_model, tokenizer, peft_model = create_peft_model(config)

train_data = combined['train']
val_data = combined['validation']
test_data = combined['test']

processed_train = processor.process_dataset(train_data)
processed_val = processor.process_dataset(val_data)
processed_test = processor.process_dataset(test_data) 

tokenized_train = processed_train.map(
    tokenize_function,
    batched=True,
    remove_columns=processed_train.column_names,
    desc="Tokenizing datasets"
)

tokenized_val = processed_val.map(
    tokenize_function,
    batched=True,
    remove_columns=processed_val.column_names,
    desc="Tokenizing datasets"
)

tokenized_test = processed_test.map(
    tokenize_function,
    batched=True,
    remove_columns=processed_test.column_names,
    desc="Tokenizing datasets"
)

tokenized_train_small = tokenized_train.filter(lambda example, index: index % 500 == 0, with_indices=True)
tokenized_val_small = tokenized_val.filter(lambda example, index: index % 500 == 0, with_indices=True)
tokenized_test_small = tokenized_test.filter(lambda example, index: index % 500 == 0, with_indices=True)

trainer = MedicalLLMTrainer(config, tokenizer, peft_model, tokenized_train, tokenized_val)

model, trainer, metrics = trainer.train_and_evaluate()
