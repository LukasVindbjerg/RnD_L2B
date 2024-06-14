""" Concat Input """

import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration, BertTokenizer, BertForSequenceClassification


def classify_order(text, bert_model, bert_tokenizer):
    inputs = bert_tokenizer(text, return_tensors="pt")
    outputs = bert_model(**inputs)
    predictions = torch.argmax(outputs.logits, dim=1)
    return "REVERSE" if predictions.item() == 0 else "FIRST"

def generate_dsl(input_text, t5_tokenizer, t5_model, order_classification):
    # Add the order classification token to the input text
    augmented_text = f"{order_classification} {input_text}"
    input_ids = t5_tokenizer.encode(augmented_text, return_tensors='pt')

    # Generate tokens
    output_ids = t5_model.generate(
        input_ids,
        max_length=50,
        num_beams=5,
        no_repeat_ngram_size=2,
        early_stopping=True,
        eos_token_id=t5_tokenizer.eos_token_id,
        pad_token_id=t5_tokenizer.pad_token_id
    )

    # Decode the generated tokens
    generated_text = t5_tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return generated_text

# Load the fine-tuned models and tokenizers
t5_tokenizer = T5Tokenizer.from_pretrained('./models/t5_label_concat_tokenizer_20240608_153342')
t5_model = T5ForConditionalGeneration.from_pretrained('./models/t5_label_concat_model_20240608_153342')
bert_tokenizer = BertTokenizer.from_pretrained('./models/bert_order_tokenizer_20240609_142648')
bert_model = BertForSequenceClassification.from_pretrained('./models/bert_order_model_20240609_142648')

# Define some test data
test_texts = ["Empty the trash", 
              "Clean the kitchen sink", 
              "Do the laundry, but first wash the car", 
              "Vacuum the living room. No never mind", 
              "Water the plants and then do the dishes"]

# Test the model
for text in test_texts:
    order_classification = classify_order(text, bert_model, bert_tokenizer)
    generated_dsl = generate_dsl(text, t5_tokenizer, t5_model, order_classification)
    print(f"Input: {text}\nOrder Classification: {order_classification}\nOutput: {generated_dsl}\n")
