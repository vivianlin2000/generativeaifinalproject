from medicaldataloader import MedicalDataLoader
from loadsymptoms import load_symptom_dataset
from medicaldataprocessor import MedicalDataProcessor

def tokenize_function(example):
    labels = tokenizer(example['response'], padding="max_length", max_length=4096, truncation=True, return_tensors="pt").to(device)

    model_inputs = tokenizer(example['instruction'], padding="max_length", max_length=4096, truncation=True, return_tensors="pt").to(device)
    model_inputs["labels"] = labels["input_ids"]

    del(labels)
    gc.collect()
    return model_inputs

def create_tokenizer(model_name):
    if model_name in ["meta-llama/Llama-2-7b-hf","meta-llama/Llama-2-13b-hf"]:
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=True)
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        return tokenizer
    elif model_name in ["google/flan-t5-small", "google/flan-t5-base"]:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        return tokenizer
    elif model_name == "facebook/opt-1.3b":
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        return tokenizer
      
loader = MedicalDataLoader()
combined = loader.create_combined_dataset()
symptoms = load_symptom_dataset()
processor = MedicalDataProcessor(include_context_markers=True)
tokenizer = create_tokenizer(model_name)

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

