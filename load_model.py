import torch
import torch.nn as nn
from transformers import GPT2Tokenizer, GPT2Config, GPT2LMHeadModel
import argparse


gpt_tokenizer = GPT2Tokenizer.from_pretrained(pretrained_model_name_or_path = "gpt2", cache_dir = '/data/bairu/model_cache/gpt2_model/gpt2/')
gpt_model = GPT2LMHeadModel.from_pretrained(pretrained_model_name_or_path = "gpt2", cache_dir = '/data/bairu/model_cache/gpt2_model/gpt2/')
gpt_config = GPT2Config.from_pretrained(pretrained_model_name_or_path = "gpt2", cache_dir = '/data/bairu/model_cache/gpt2_model/gpt2/')

# print("param size: ", gpt_model.transformer.wte.weight.size())
# print("config vocab: ", gpt_config.vocab_size)
# print("tokenizer size: ", len(gpt_tokenizer.encoder), len(gpt_tokenizer.byte_encoder))

gpt_tokenizer.save_pretrained('/data/bairu/model_cache/saved_gpt2/gpt2/')
gpt_model.save_pretrained('/data/bairu/model_cache/saved_gpt2/gpt2/')
gpt_config.save_pretrained('/data/bairu/model_cache/saved_gpt2/gpt2/')

