""" Concat the input embeddings with the order classification embeddings """

from transformers import BertModel, T5ForConditionalGeneration, T5Tokenizer, BertTokenizer
import torch

# Load fine-tuned models and tokenizers
bert_model = BertModel.from_pretrained('./models/bert_order_model')
bert_tokenizer = BertTokenizer.from_pretrained('./models/bert_order_tokenizer')
t5_model = T5ForConditionalGeneration.from_pretrained('./models/t5_custom_vocab_model')
t5_tokenizer = T5Tokenizer.from_pretrained('./models/t5_custom_vocab_tokenizer')

def generate_dsl(input_text, order_text, t5_tokenizer, t5_model, bert_tokenizer, bert_model):
    # Encode the input text using T5 tokenizer
    input_ids = t5_tokenizer.encode(input_text, return_tensors='pt')
    input_embeds = t5_model.encoder(input_ids).last_hidden_state

    # Encode the order text using BERT tokenizer
    order_inputs = bert_tokenizer(order_text, return_tensors='pt')
    order_embeds = bert_model(**order_inputs).last_hidden_state

    # Combine embeddings
    combined_embeds = torch.cat((input_embeds, order_embeds), dim=1)

    # Generate tokens using the T5 decoder with combined embeddings
    output_ids = t5_model.generate(
        inputs_embeds=combined_embeds,
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

# Example usage
input_text = "Do the laundry, but first wash the car"
order_text = "REVERSE"  # This is just an example; in practice, you would generate this using the BERT model
generated_dsl = generate_dsl(input_text, order_text, t5_tokenizer, t5_model, bert_tokenizer, bert_model)
print(f"Input: {input_text}\nOrder Classification: {order_text}\nOutput: {generated_dsl}")
