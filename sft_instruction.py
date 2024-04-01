import os
import json

from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from trl import SFTTrainer
from transformers import DataCollatorForLanguageModeling, EarlyStoppingCallback


# file_path = 'drive/MyDrive/SNLP/emotion_alpaca_v4_cleaned.json'
file_path = 'COMP0087_SNLP/alpaca-instruction/emotion_alpaca_v4_cleaned.json'

with open(file_path, 'r') as file:
    data = json.load(file)
    
dataset = Dataset.from_dict({
    'instruction': [item['instruction'] for item in data],
    'input': [item['input'] for item in data],
    'output': [item['rewritten_output'] for item in data]
})

dataset = dataset.train_test_split(test_size=0.1)

tokenizer = AutoTokenizer.from_pretrained("gpt2-medium")
tokenizer.pad_token = tokenizer.eos_token

# def formatting_func(batch):
#     # Process each example in the batch and return a list of formatted strings
#     return [
#         f"{tokenizer.bos_token} Instruction: {instr} Input: {inp} Output: {out} {tokenizer.eos_token}"
#         for instr, inp, out in zip(batch['instruction'], batch['input'], batch['output'])
#     ]

def formatting_func(batch):
    formatted_example = []
    for instr, inp, out in zip(batch['instruction'], batch['input'], batch['output']):
        format_prompt = f"{tokenizer.bos_token} Write an appropriate response to the instruction."
        if inp:
            format_prompt += f"Instruction: {instr}\nInput: {inp}\nResponse: {out}{tokenizer.eos_token}"
        else:
            format_prompt += f"Instruction: {instr}\nResponse: {out}{tokenizer.eos_token}"
        formatted_example.append(format_prompt)
    return formatted_example
    
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

model = AutoModelForCausalLM.from_pretrained("gpt2-medium")

LR = 2e-5            # Learning rate was 5e-5
PATIENCE = 10        # Patience for early stopping
BSZ = 4              # Batch size
EVAL_EVERY = 200     # Evaluate every X steps
SAVE_EVERY = 200     # Save model checkpoint every X steps
MAX_EPOCHS = 10      # Maximum number of epochs

# Define training arguments
training_args = TrainingArguments(
    output_dir="new_gpt2_finetuned_instruction",
    evaluation_strategy="steps",
    learning_rate=LR,
    per_device_train_batch_size=BSZ,
    per_device_eval_batch_size=BSZ,
    num_train_epochs=MAX_EPOCHS,
    eval_steps=EVAL_EVERY,
    save_steps=SAVE_EVERY,
    load_best_model_at_end=True,
    gradient_accumulation_steps=8, #was 2
    fp16=True,
    lr_scheduler_type="cosine",
)

# Initialize SFTTrainer
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    args=training_args,
    train_dataset=dataset['train'],
    eval_dataset=dataset['test'],
    data_collator=data_collator,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=PATIENCE)],
    max_seq_length=1024,
    formatting_func=formatting_func
)

# Start training
trainer.train()

model.save_pretrained("./new_model-medium_folder")
tokenizer.save_pretrained("./new_model-medium_folder")