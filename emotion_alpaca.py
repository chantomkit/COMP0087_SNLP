import os
import torch
import random
import pickle
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, TextStreamer
from datasets import load_dataset
from accelerate.utils import release_memory
from tqdm import tqdm


# get token and device
token = os.environ.get("HF_TOKEN", None)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# load model
print("Loading model...")
base_model = "mistralai/Mistral-7B-Instruct-v0.2"
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

# create prompt
opening = [
    "Please rewrite the [Original Output] to convey the emotion of [Desired Emotion] while maintaining the core message and intent of the [Instruction] and [Input].",
    "Here are the requirements: "
]
rules = [
    "There is no need to remember the conversation history except this prompt. The history prompts are independent.",
    "Your response should be in exactly one paragraph with simple children level language.",
    "Your response should be highly related to the emotion and context without too much plot twist",
    "Your response should not explain the context behind your generation",
    "Not all instructions require input. For example, when a instruction asks about some general information, \"what is the highest peak in the world\", it is not necessary to provide a specific context. In this case, we simply put \"\" in the input field."
]
examples = [
    "For example:",
    "Instruction: Come up with a creative recipe for breakfast.",
    "Input: """,
    "Desired Emotion: joy",
    "Original Output: French toast filled with Nutella and fresh strawberries, finished with a topping of whipped cream and a drizzle of chocolate sauce. This is a rich and sweet option for breakfast.",
    "Rewritten Output: French toast stuffed with Nutella and fresh strawberries, topped with whipped cream and drizzled with chocolate sauce. A sweet and decadent way to start your day!",
]
ending = [
    "Now rewrite the output based on the emotions and context:",
]
instruction_prompt = "\n".join(opening + rules + examples + ending)

# auxiliary function
positive_tones = ['admiration', 'amusement', 'approval', 'caring',
                      'curiosity', 'desire', 'excitement', 'gratitude',
                      'joy', 'love', 'optimism', 'pride', 'realization',
                      'relief', 'surprise']

def generate_message(emotion, context, instruction_prompt=instruction_prompt, return_chat_template=False):
    original_instruction = f"Instruction: {context['instruction']}"
    original_input = f"Input: {context['input']}"
    desired_emotion = f"Desired Emotion: {emotion}"
    original_output = f"Original Output: {context['output']}"
    # task_prompt = f"Desired emotion: {emotion}: {context} => "
    task_prompt = original_instruction + "\n" + original_input + "\n" + desired_emotion + "\n" + original_output
    if return_chat_template:
        return [
            {"role": "user", "content": instruction_prompt + "\n" + task_prompt},
        ]
    return "[INST]" + instruction_prompt + "\n" + task_prompt + "[/INST]"

def generate_response(messages):
    model.eval()
    inputs = tokenizer(messages, return_tensors="pt", padding=True).to("cuda")
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=512,
            temperature=0.7,
            top_k=0,
            top_p=0.9,
            repetition_penalty=1,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
    decodeds = tokenizer.batch_decode(output)
    inputs, output = release_memory(inputs, output)
    release_memory(model)
    return decodeds

def convey_emo(ds, batch_size=32):
  augment_ds = []
  instructions, inputs, emotions, messages = [], [], [], []
  for i, d in enumerate(tqdm(ds)):
    instructions.append(d['instruction'])
    inputs.append(d['input'])
    emotions.append(random.choice(positive_tones))
    message = generate_message(emotions[-1], d)
    messages.append(message)
    if ((i+1) % batch_size == 0) or ((i+1) >= (len(ds) - (len(ds) % batch_size))):
      responses = generate_response(message)
      responses = [r.split("[/INST]")[1].replace("</s>", "").strip() for r in responses]
      augment_ds += [
          {
              "instruction": ins,
              "input": inp,
              "emotion": e,
              "new-output": out
          }
          for (ins, inp, e, out) in zip(instructions, inputs, emotions, responses)
      ]
      # clear the history
      instructions, inputs, emotions, messages = [], [], [], []

  return augment_ds

# main function
def main():
  augment_ds = convey_emo(dataset['train'], batch_size=32)
  print("Saving the augmented dataset...")
  save_path = "emotion_alpaca.pkl"
  with open(save_path, 'wb') as file:
    pickle.dump(augment_ds, file)
    
if __name__ == "__main__":
  main()