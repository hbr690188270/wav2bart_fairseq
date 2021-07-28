import torch
import torch.nn as nn
from transformers import GPT2Tokenizer, GPT2Config, GPT2LMHeadModel
from fairseq.models.wav2bart.gpt2_decoder import transformer_lm_gpt,base_lm_architecture
import argparse
from fairseq.models.transformer import TransformerDecoder

parser = argparse.ArgumentParser()
parser.add_argument("--test", default = '')
args = parser.parse_args()

gpt_model_args = transformer_lm_gpt(args)

gpt_model = TransformerDecoder(args,dictionary = {'<s>':0},
                embed_tokens = nn.Embedding(num_embeddings = 1, embedding_dim = 512), no_encoder_attn=False,)

f = open("./gpt2_params.txt",'w', encoding = 'utf-8')
for name, param in gpt_model.named_parameters():
    print(name + '\n', file = f)
    print(name)
f.close()
hg_gpt2 = GPT2LMHeadModel.from_pretrained(model_tpye = "gpt_small", cache_dir = '/data/bairu/model_cache/gpt2/gpt2_small/')

f = open("./hg_gpt2_params.txt",'w', encoding = 'utf-8')

for name, param in hg_gpt2.named_parameters():
    print(name + '\n', file = f)
    print(name)
f.close()
