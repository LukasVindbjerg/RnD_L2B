from transformers import T5Tokenizer, T5ForConditionalGeneration, Trainer, TrainingArguments
import pandas as pd
import torch
import csv

# Load pre-trained T5 model and tokenizer
model_name = "t5-small"
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)

# Load the custom vocabulary from the CSV file
vocab_path = 'data/custom_vocab.csv'
custom_vocab = []
with open(vocab_path, 'r') as file:
    reader = csv.reader(file)
    next(reader)  # Skip the header
    for row in reader:
        custom_vocab.append(row[0])

# Add essential tokens to ensure tokenizer functionality
essential_tokens = ['[PAD]', '[EOS]', '[SEP]', '[CLS]', '[MASK]']
full_vocab = essential_tokens + custom_vocab

# Ensure no duplicates
full_vocab = list(set(full_vocab))

# Recreate the tokenizer with only the custom vocabulary
tokenizer.add_tokens(full_vocab)
model.resize_token_embeddings(len(tokenizer))

# Encode the special tokens manually
special_tokens_dict = {'pad_token': '[PAD]', 'eos_token': '[EOS]'}
tokenizer.add_special_tokens(special_tokens_dict)

# Load the training data
# df = pd.read_csv('data/dsl_training.csv')
df = pd.read_csv('data/train_data_test.csv')
train_texts = df['input'].tolist()
train_labels = df['output'].tolist()

# Encode the training data with padding
train_encodings = tokenizer(train_texts, padding=True, truncation=True, return_tensors="pt")
label_encodings = tokenizer(train_labels, padding=True, truncation=True, return_tensors="pt")

# Shift the labels to the right to create the labels tensor
labels = label_encodings['input_ids'].clone()
labels[labels == tokenizer.pad_token_id] = -100  # Ignore padding tokens in loss computation

class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels
    
    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item
    
    def __len__(self):
        return len(self.labels)

train_dataset = CustomDataset(train_encodings, labels)

# Define training arguments
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=2,
    per_device_train_batch_size=2,
    warmup_steps=10,
    weight_decay=0.01,
    logging_dir='./logs',
)

# Initialize the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
)

# Fine-tune the T5 model
trainer.train()

# Save the fine-tuned model and tokenizer
model.save_pretrained('./models/t5_custom_vocab_model')
tokenizer.save_pretrained('./models/t5_custom_vocab_tokenizer')


