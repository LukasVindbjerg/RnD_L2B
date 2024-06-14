from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments, EarlyStoppingCallback
import pandas as pd
import torch
import csv
import datetime
from sklearn.preprocessing import LabelEncoder

# Get current date and time
now = datetime.datetime.now()
timestamp = now.strftime("%Y%m%d_%H%M%S")

# Load pre-trained BERT model and tokenizer
bert_model_name = "bert-base-uncased"
bert_tokenizer = BertTokenizer.from_pretrained(bert_model_name)
bert_model = BertForSequenceClassification.from_pretrained(bert_model_name, num_labels=6) 

# Load the training data for BERT
df_order = pd.read_csv('data/full_dataset.csv')
train_texts = df_order['text'].tolist()
train_labels = df_order['label'].tolist()  

label_encoder = LabelEncoder()
train_labels_encoded = label_encoder.fit_transform(train_labels)

# Encode the training data
order_encodings = bert_tokenizer(train_texts, padding=True, truncation=True, return_tensors="pt")
train_labels_encoded = torch.tensor(train_labels_encoded)

class OrderDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels
    
    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item
    
    def __len__(self):
        return len(self.labels)

dataset = OrderDataset(order_encodings, train_labels_encoded)

# Split the dataset into training and evaluation sets
train_size = int(0.8 * len(dataset))
eval_size = len(dataset) - train_size
train_dataset, eval_dataset = torch.utils.data.random_split(dataset, [train_size, eval_size])

# Calculate total training steps and warmup steps
batch_size = 8  # Adjust based on your GPU memory
epochs = 50  # Increase the number of epochs for better training
total_steps = (len(train_dataset) * epochs) // batch_size
warmup_steps = int(0.1 * total_steps)  # 10% of total steps
# Define training arguments for BERT


bert_training_args = TrainingArguments(
    output_dir=f'./results_order_{timestamp}',
    num_train_epochs=epochs,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,  # Same as training batch size
    warmup_steps=warmup_steps,
    weight_decay=0.01,
    logging_dir=f'./logs/logs_order_{timestamp}',
    report_to="tensorboard",  # Report to TensorBoard
    logging_steps=10,  # Log every 10 steps
    eval_strategy="steps",  # Use eval_strategy instead of evaluation_strategy
    eval_steps=100,  # Evaluate every 500 steps
    save_steps=500,  # Save checkpoint every 500 steps
    save_total_limit=3,  # Save up to 3 checkpoints
    load_best_model_at_end=True,  # Load the best model found during evaluation
    metric_for_best_model="loss",  # Use loss to determine the best model
    greater_is_better=False,  # Lower loss is better
)

# Initialize the Trainer for BERT
bert_trainer = Trainer(
    model=bert_model,
    args=bert_training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,  # Add evaluation dataset
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]  # Add TensorBoard callback
)

# Fine-tune the BERT model
bert_trainer.train()

# Save the fine-tuned BERT model and tokenizer
bert_model.save_pretrained(f'./models/bert_order_model_{timestamp}')
bert_tokenizer.save_pretrained(f'./models/bert_order_tokenizer_{timestamp}')