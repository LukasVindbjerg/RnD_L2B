from transformers import GPT2Tokenizer, GPT2LMHeadModel
import csv
import pandas as pd
import torch
from transformers import DataCollatorForLanguageModeling, Trainer, TrainingArguments

# Load pre-trained model and tokenizer
model_name = "gpt2"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)

# Load the custom vocabulary from the CSV file
# vocab_path = 'data/custom_vocab.csv'
vocab_path = 'data/custom_vocab.csv'

custom_vocab = []
with open(vocab_path, 'r') as file:
    reader = csv.reader(file)
    next(reader)  # Skip the header
    for row in reader:
        custom_vocab.append(row[0])

essential_tokens = ['[', ']', '(', ')', '.', '[PAD]']
for token in essential_tokens:
    if token not in custom_vocab:
        custom_vocab.append(token)


# Add custom tokens to the tokenizer
tokenizer.add_tokens(custom_vocab)

# Add a padding token to the tokenizer
tokenizer.add_special_tokens({'pad_token': '[PAD]'})

# Resize the model embeddings to match the new tokenizer size
model.resize_token_embeddings(len(tokenizer))

# Load the CSV file using pandas
# df = pd.read_csv('data/dsl_training.csv')
df = pd.read_csv('data/dsl_training.csv')

# Extract the input and output columns
train_texts = df['input'].tolist()
train_labels = df['output'].tolist()

# Encode the training data with padding
train_encodings = tokenizer(train_texts, return_tensors='pt', padding=True, truncation=True)
# Encode the labels
label_encodings = tokenizer(train_labels, return_tensors='pt', padding=True, truncation=True)

# Create attention masks
train_attention_masks = train_encodings['attention_mask']
label_attention_masks = label_encodings['attention_mask']

# Shift the labels to the right to create the labels tensor
labels = label_encodings['input_ids'].clone()
labels[labels == tokenizer.pad_token_id] = -100  # Ignore padding tokens in loss computation

class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels, attention_masks):
        self.encodings = encodings
        self.labels = labels
        self.attention_masks = attention_masks
    
    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = self.labels[idx]
        item['attention_mask'] = self.attention_masks[idx]
        return item
    
    def __len__(self):
        return len(self.encodings['input_ids'])

train_dataset = CustomDataset(train_encodings, labels, train_attention_masks)

# Define data collator to handle padding dynamically
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,  # We are not doing masked language modeling
)

training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=10,
    per_device_train_batch_size=2,
    warmup_steps=10,
    weight_decay=0.01,
    logging_dir='./logs',
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    data_collator=data_collator,
)

trainer.train()

# Save the fine-tuned model and tokenizer
model.save_pretrained('./models/custom_vocab_model_test')
tokenizer.save_pretrained('./models/custom_vocab_tokenizer_test')

# Load the fine-tuned model and tokenizer
model = GPT2LMHeadModel.from_pretrained('./models/custom_vocab_model_test')
tokenizer = GPT2Tokenizer.from_pretrained('./models/custom_vocab_tokenizer_test')


def generate_custom_text(input_text, tokenizer, model, custom_tokens):
    input_ids = tokenizer.encode(input_text, return_tensors='pt')
    
    # Create attention mask
    attention_mask = input_ids.ne(tokenizer.pad_token_id).long()
    
    # Generate tokens with the model
    output_ids = model.generate(input_ids, max_length=50, attention_mask=attention_mask)
    
    # Decode the generated tokens
    generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    
    # Ensure the generated text uses only custom vocabulary
    tokens = generated_text.split()
    filtered_tokens = [token for token in tokens if token in custom_tokens]
    
    # Join filtered tokens to form the final output
    final_output = ' '.join(filtered_tokens)
    return final_output
# Define some test data
test_texts = ["Empty the trash", "Clean the kitchen sink"]

# Test the model with the custom generation function
for text in test_texts:
    generated_text = generate_custom_text(text, tokenizer, model, custom_vocab)
    print(f"Input: {text}\nOutput: {generated_text}\n")

