# this is the main file of the project, which fine-tunes the llama 8B model to be a classifier
# dataset: PRM800K

# input: instruction, responses, next response and the answer
# output: predicted label (0, 1 or 2), where 0 means wrong, 1 means ambiguous, and 2 means correct.

# the structure of this file: preprocess the dataset, fine-tune the model, and evaluate the performance on the test set.

# login to the Hugging Face Hub
my_api_key = "hf_..."
from huggingface_hub import notebook_login
notebook_login(my_api_key)

# Load the tokenizer and model
from transformers import AutoTokenizer, AutoModelForCausalLM
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3.1-8B")
model = AutoModelForCausalLM.from_pretrained("meta-llama/Meta-Llama-3.1-8B")

# load dataset
from datasets import load_dataset
dataset = load_dataset("Birchlabs/openai-prm800k-stepwise-critic")

# some preprocessing
# Define a prompt template
prompt = "{}\n{}\n{}"

# Define an EOS (End Of Sequence) token from the tokenizer
EOS_TOKEN = tokenizer.eos_token

# Define a fixed instruction for the classification task
instruction = "Predict the rating of the 'Next Response' based on the following information, where 2 means correct, 1 means ambiguous, and 0 means wrong."

# Define the function to format the prompts
def formatting_prompts_func(examples):
    inputs = []
    outputs = examples["rating"]

    # Iterate through each example in the list of examples
    for i in range(len(examples["instruction"])):
        # Combine all fields except 'rating' into the input string
        input_str = f"Instruction: {examples['instruction'][i]}\n"
        input_str += f"Responses: {examples['responses'][i]}\n"
        input_str += f"Next Response: {examples['next_response'][i]}\n"
        input_str += f"Answer: {examples['answer'][i]}\n"
        input_str += EOS_TOKEN  # Add EOS token at the end
        inputs.append(input_str)

    # Format the text using the prompt template
    formatted_texts = [prompt.format(instruction, input_text, "") for input_text in inputs]

    return {"text": formatted_texts, "output": outputs}

# Apply the formatting function to the dataset.
# Now the 'text' and 'output' fields are all we need
dataset = dataset.map(formatting_prompts_func, batched=True)

print(dataset['train'][0])

# Define the function to tokenize the input text
def tokenize_func(examples):
    return tokenizer(examples["text"], padding=True, truncation=True)

# Apply the tokenization function to the dataset
dataset = dataset.map(tokenize_func, batched=True)

# Import necessary modules for training
import torch
from torch.utils.data import DataLoader
from transformers import Trainer, TrainingArguments

# Define the labels as a dictionary to map the label values to indices
label_to_id = {-1: 0, 0: 1, 1: 2}

# Map the labels to their corresponding indices
def label_mapping_func(examples):
    examples["labels"] = [label_to_id[label] for label in examples["output"]]
    return examples

# Apply the label mapping function to the dataset
dataset = dataset.map(label_mapping_func, batched=True)

# Define the data collator to handle the padding and batch processing
from transformers import DataCollatorWithPadding
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# Split the dataset into train and validation sets
train_dataset = dataset['train']
val_dataset = dataset['validation']

# Create data loaders for training and validation
train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=8, collate_fn=data_collator)
val_dataloader = DataLoader(val_dataset, batch_size=8, collate_fn=data_collator)

# Define the training arguments
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=10,
    save_total_limit=2,
    load_best_model_at_end=True,
)

# Define the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    data_collator=data_collator,
    tokenizer=tokenizer,
)

# Start training the model
trainer.train()

# Save the fine-tuned model
trainer.save_model("./fine-tuned-llama")

# Example of how to use the trained model for inference
def predict_rating(text):
    inputs = tokenizer(text, return_tensors="pt").to("cuda")
    outputs = model.generate(**inputs, max_new_tokens=64, use_cache=True)
    prediction = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    return prediction

# Example usage
sample_text = dataset['validation'][0]['text']
predicted_rating = predict_rating(sample_text)
print(f"Predicted Rating: {predicted_rating}")

# accuracy evaluation on the whole test set
from sklearn.metrics import accuracy_score

test_dataset = dataset['validation']
test_dataloader = DataLoader(test_dataset, batch_size=8, collate_fn=data_collator)

model.to("cuda")
model.eval()

all_predictions = []
all_labels = []

for batch in test_dataloader:
    inputs = {k: v.to("cuda") for k, v in batch.items() if k != "labels"}
    with torch.no_grad():
        outputs = model(**inputs)
        predictions = torch.argmax(outputs.logits, dim=-1)
    all_predictions.extend(predictions.cpu().tolist())
    all_labels.extend(batch["labels"])

accuracy = accuracy_score(all_labels, all_predictions)
print(f"Test Accuracy: {accuracy}")
