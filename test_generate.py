import torch
from transformers import AutoTokenizer

from models.modeling_luduan import LuduanForCausalLM
from models.configuration_luduan import LuduanConfig


import argparse

def parse_arguments():
    parser = argparse.ArgumentParser(description='Test')

    parser.add_argument('--checkpoint_file', type=str, required=True, help='input filename')
    parser.add_argument('--tokenizer_name', type=str, required=True, help='output filename')
    parser.add_argument('--text', type=str, required=True, help='output filename')

    
    return parser.parse_args()

args = parse_arguments()

nano_luduan_config = LuduanConfig(
    vocab_size=64000,
    n_embd=768,
    n_layer=12,
    n_head=12,
    block_size=1024,
    intermediate_size=768 * 4)

model = LuduanForCausalLM(nano_luduan_config)
tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name, trust_remote_code=True)
model.load_state_dict(torch.load(args.checkpoint_file))
model.to('cuda:0')

text = args.text
encoded_input = tokenizer(text, return_tensors='pt').to('cuda:0')
pred = model.generate(**encoded_input, max_new_tokens=64,repetition_penalty=1.1)
print(tokenizer.decode(pred.cpu()[0], skip_special_tokens=True))
