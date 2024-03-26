import os
import torch
import random
import numpy as np
import pickle
import utils
import json
import re
import time
import fire
import tqdm
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, TextStreamer
from datasets import load_dataset
from accelerate.utils import release_memory



# get token and device
token = os.environ.get("HF_TOKEN", None)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# load model
print("Loading model...")
base_model = "mistralai/Mistral-7B-Instruct-v0.1"
config = AutoConfig.from_pretrained(base_model, trust_remote_code=True)
config.max_position_embeddings = 8096
quantization_config = BitsAndBytesConfig(
    llm_int8_enable_fp32_cpu_offload=True,
    bnb_4bit_quant_type='nf4',
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    load_in_4bit=True,
    )

tokenizer = AutoTokenizer.from_pretrained(base_model, use_fast=True, token=token)
model = AutoModelForCausalLM.from_pretrained(
    base_model,
    config=config,
    trust_remote_code=True,
    quantization_config=quantization_config,
    device_map="auto",
    offload_folder="./offload",
    token = token
    )
tokenizer.pad_token = tokenizer.eos_token

# load the dataset
print("Loading dataset...")
dataset = load_dataset("yahma/alpaca-cleaned")

def line_processor(line):
    line = re.sub("[\t\n]", "", line) # remove tabs and newlines
    line = re.sub(r'\s+([.,!?;:])', r'\1', line) # remove spaces before punctuation
    line = line.strip() # remove leading and trailing spaces
    if len(line.split()) <= 10: # remove lines with less than 10 words
        return None
    return line

def unpack_response(response):
    response_text = response[0]
    # Extract the relevant parts from the response
    start_index = response_text.rfind("Instruction: ") + len("Instruction: ")
    end_index = response_text.rfind("[/INST]")
    relevant_text = response_text[start_index:end_index]
    
    instruction = relevant_text.split("\nInput:")[0]
    input_text = relevant_text.split("\nInput:")[1].split("\nDesired Emotion:")[0].strip()
    emotion = relevant_text.split("Desired Emotion: ")[1].split("\nOriginal Output:")[0]
    original_output = relevant_text.split("Original Output: ")[1].split("\n[/INST]")[0].strip()
    rewritten_output = relevant_text.split("Rewritten Output: ")[1].strip()
    rewritten_output = line_processor(rewritten_output)
    
    return {
        "instruction": instruction,
        "input": input_text,
        "emotion": emotion,
        "original_output": original_output,
        "rewritten_output": rewritten_output
    }

# create prompt
def encode_prompt(seed_instructions, prompt_instructions, emotion):
    prompt = open('./emotion_alpaca_prompt_v4.txt', 'r').read() + "\n\n" # load the prompt
    # Add specific instructions to the prompt
    prompt += "Important Guidelines:\n"
    prompt += "- The rewritten output should maintain the core meaning and facts from the original output.\n"
    prompt += "- Do not introduce new information or change the context of the original output.\n"
    prompt += "- Focus on conveying the desired emotion through the tone, style, and choice of words, while preserving the original message.\n"
    prompt += "- If the desired emotion does not align with the original output, aim to express in an appropriate positive emotion without altering the fundamental meaning.\n"
    prompt += "- Even for factual or straightforward instructions, try to incorporate emotional language or phrases to convey the desired emotion.\n\n"
    prompt += "- Use simple, age-appropriate language suitable for children aged 8-12 years old.\n"
    prompt += "- Employ shorter sentences and avoid complex sentence structures to enhance clarity and understanding.\n\n"

    for i, task in enumerate(seed_instructions, start=1):
        instruction = task['instruction']
        instruction = re.sub(r"\s+", " ", instruction).strip().rstrip(":")
        input_data = task['input']
        input_data = "<noinput>" if input_data.lower() == "" else input_data
        old_output = task['old_output']
        new_output = task['new_output']

        prompt += f"Example {i}:\n"
        prompt += f"Instruction: {instruction}\n"
        prompt += f"Input: {input_data}\n"
        prompt += f"Desired Emotion: {emotion}\n"
        prompt += f"Original Output: {old_output}\n"
        prompt += f"Rewritten Output: {new_output}\n\n"
        
    prompt += "Please generate the rewritten output and respond with only the modified text, without any additional labels, tags, prefixes, or explanations. The rewritten output should be a direct response to the [Instruction] and [Input], conveying the [Desired Emotion] in a concise and appropriate manner.\n"
    prompt += "Remember to use simple, child-friendly language and shorter sentences to ensure clarity and understanding for children aged 8-12 years old.\n\n"
    
    for idx, task_dict in enumerate(prompt_instructions):
        # print(task_dict)
        (instruction, input, output) = task_dict["instruction"], task_dict["input"], task_dict["output"]
        # clean the sentences
        instruction = re.sub(r"\s+", " ", instruction).strip().rstrip(":")
        input = "<noinput>" if input.lower() == "" else input
        prompt += f"[INST]"
        prompt += f"Instruction: {instruction}\n"
        prompt += f"Input:{input}\n"
        prompt += f"Desired Emotion: {emotion}\n"
        prompt += f"Original Output:{output}"
        prompt += "[/INST]\n"
    
    # print(prompt)
    return prompt

def generate_response(prompts):
    model.eval()
    inputs = tokenizer(prompts, return_tensors="pt", padding=True).to("cuda")
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=512,
            temperature=0.8,
            top_p=0.95,
            repetition_penalty=1.3,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
    responses = tokenizer.batch_decode(output)
    inputs, output = release_memory(inputs, output)
    release_memory(model)
    return responses

# now let's generate new alpaca instructions!
def emo_alpaca(
    dataset=dataset["train"],
    output_dir = "./",
    seed_task_path = "./emo_seed_task.json",
    start_idx = 0,
    num_instructions_to_generate = 100,
    request_batch_size = 5,
): 
    # set a seed
    random.seed(42)
    with open(seed_task_path, 'r') as file:
        emo_seed_tasks = json.load(file)
    print(f"Loaded emotion seed tasks")
    
    os.makedirs(output_dir, exist_ok=True)
    request_idx = 0
    # load the generated_instructions
    emotion_data = []
    if os.path.exists(os.path.join(output_dir, "emotion_alpaca_v4.json")):
        emotion_data = utils.jload(os.path.join(output_dir, "emotion_alpaca_v4.json"))
        print(f"Loaded {len(emotion_data)} generated instructions")
    
    # positive emotions
    positive_tones = ['admiration', 'amusement', 'approval', 'caring',
                        'curiosity', 'desire', 'excitement', 'gratitude',
                        'joy', 'love', 'optimism', 'pride', 'realization',
                        'relief', 'surprise']
    
    progress_bar = tqdm.tqdm(total=num_instructions_to_generate)
    if emotion_data:
        progress_bar.update(len(emotion_data))
        start_idx = len(emotion_data)
        if num_instructions_to_generate == -1:
            # generate all the instructions
            num_instructions_to_generate = len(dataset) - len(emotion_data)
        
    for idx in range(start_idx, start_idx + num_instructions_to_generate, request_batch_size):
        request_idx += 1
        batch_inputs = []
        random_emotion = np.random.choice(positive_tones,request_batch_size,replace=False)
        
        for i in range(request_batch_size):
            seed_instructions = emo_seed_tasks[random_emotion[i]] # this is already a list
            sampled_seed_instructions = random.sample(seed_instructions, 3)
            prompt_instructions = [dataset[idx + i]]
            prompt = encode_prompt(sampled_seed_instructions, prompt_instructions, random_emotion[i])
            batch_inputs.append(prompt)
        
        request_start = time.time()
        responses = generate_response(batch_inputs)
        request_duration = time.time() - request_start
        
        responses = [r.split("[/INST]")[1].replace("</s>", "").strip() for r in responses]
        
        for r in range(len(responses)):
            emotion_data.append(
                {'instruction': dataset[idx + r]['instruction'],
                    'input': dataset[idx + r]['input'],
                    # 'emotion': random_emotion[r],
                    # 'original_output': dataset[idx + r]['output'],
                    'rewritten_output': responses[r]
                }
            )
            progress_bar.update(1)
        
        print(f"Request {request_idx} took {request_duration:.2f} seconds")
        utils.jdump(emotion_data, os.path.join(output_dir, "emotion_alpaca_v4.json"))
        
# main function
def main(task, **kwargs):
    globals()[task](**kwargs)
    
if __name__ == "__main__":
    fire.Fire(main)