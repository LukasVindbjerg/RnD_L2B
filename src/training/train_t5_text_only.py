import pandas as pd
import torch
import csv
import datetime
from transformers import T5Tokenizer, T5ForConditionalGeneration, Trainer, TrainingArguments, EarlyStoppingCallback
from transformers.integrations import TensorBoardCallback

# Get current date and time
now = datetime.datetime.now()
timestamp = now.strftime("%Y%m%d_%H%M%S")

# Load pre-trained T5 model and tokenizer
model_name = "t5-small"
tokenizer = T5Tokenizer.from_pretrained(model_name, legacy=False)
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
model.resize_token_embeddings(len(tokenizer))  # Resize token embeddings to accommodate new tokens

# Encode the special tokens manually
special_tokens_dict = {'pad_token': '[PAD]', 'eos_token': '[EOS]'}
tokenizer.add_special_tokens(special_tokens_dict)

# Load the training data
df = pd.read_csv('data/full_dataset.csv')

# Filter out rows with empty 'dsl' values
df = df[df['dsl'].notna()]

train_texts = df['text'].tolist()
train_labels = df['dsl'].tolist()

# Ensure train_labels are strings
train_labels = [str(label) for label in train_labels]

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
        item = {key: val[idx].clone().detach() for key, val in self.encodings.items()}
        item['labels'] = self.labels[idx].clone().detach()
        return item
    
    def __len__(self):
        return len(self.labels)

# Create the dataset
dataset = CustomDataset(train_encodings, labels)

# Split the dataset into training and evaluation sets
train_size = int(0.8 * len(dataset))
eval_size = len(dataset) - train_size
train_dataset, eval_dataset = torch.utils.data.random_split(dataset, [train_size, eval_size])

# Calculate total training steps and warmup steps
batch_size = 8  # Adjust based on your GPU memory
epochs = 50  # Increase the number of epochs for better training
total_steps = (len(train_dataset) * epochs) // batch_size
warmup_steps = int(0.1 * total_steps)  # 10% of total steps

# Define training arguments
training_args = TrainingArguments(
    output_dir=f'./results_text_only_{timestamp}',  # Directory to save model checkpoints
    num_train_epochs=epochs,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,  # Same as training batch size
    warmup_steps=warmup_steps,
    weight_decay=0.01,
    logging_dir=f'./logs/text_only_{timestamp}',  # Directory to save the logs
    report_to="tensorboard",  # Report to TensorBoard
    logging_steps=10,  # Log every 10 steps
    eval_strategy="steps",  # Use eval_strategy instead of evaluation_strategy
    eval_steps=500,  # Evaluate every 500 steps
    save_steps=500,  # Save checkpoint every 500 steps
    save_total_limit=3,  # Save up to 3 checkpoints
    load_best_model_at_end=True,  # Load the best model found during evaluation
    metric_for_best_model="loss",  # Use loss to determine the best model
    greater_is_better=False,  # Lower loss is better
)

# Initialize the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,  # Add evaluation dataset
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]  # Add TensorBoard callback
)

# Fine-tune the T5 model
trainer.train()

# Save the fine-tuned model and tokenizer
model.save_pretrained(f'./models/t5_text_only_model_{timestamp}')
tokenizer.save_pretrained(f'./models/t5_text_only_tokenizer_{timestamp}')
