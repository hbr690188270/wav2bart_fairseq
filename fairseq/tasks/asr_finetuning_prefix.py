# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

import logging
import os
import sys
import torch

from argparse import Namespace
from dataclasses import dataclass, field
from typing import Optional, Any
from omegaconf import MISSING, II, OmegaConf

from fairseq.data import (
    BinarizedAudioDataset,
    Dictionary,
    FileAudioDataset,
    encoders,
)
from fairseq.dataclass import FairseqDataclass
from fairseq.dataclass.configs import GenerationConfig
from fairseq.data import AddTargetDatasetBart as AddTargetDataset

from . import FairseqTask, register_task
from .. import utils
from ..logging import metrics

import pickle
import numpy as np
from fairseq.models.transformer_lm import TransformerLanguageModel

logger = logging.getLogger(__name__)


class LabelEncoder(object):
    def __init__(self, dictionary):
        self.dictionary = dictionary

    def __call__(self, label):
        id_list = self.dictionary.convert_tokens_to_ids(
            label
        )
        return torch.IntTensor(id_list)



@dataclass
class ASRFinetuningPrefixConfig(FairseqDataclass):
    data: str = field(default=MISSING, metadata={"help": "path to data directory"})
    labels: Optional[str] = field(
        default=None,
        metadata={"help": "extension of the label file to load, used for fine-tuning"},
    )
    sample_rate: int = field(
        default=16_000,
        metadata={
            "help": "target sample rate. audio files will be up/down sampled to this rate"
        },
    )
    normalize: bool = field(
        default=False,
        metadata={"help": "if set, normalizes input to have 0 mean and unit variance"},
    )
    enable_padding: bool = field(
        default=False, metadata={"help": "pad shorter samples instead of cropping"}
    )
    max_sample_size: Optional[int] = field(
        default=None, metadata={"help": "max sample size to crop to for batching"}
    )
    min_sample_size: Optional[int] = field(
        default=None, metadata={"help": "min sample size to skip small examples"}
    )

    # Options for reporting WER metrics during validation. Only applicable to
    # Seq2Seq models during fine-tuning
    eval_wer: bool = field(
        default=False, metadata={"help": "compute WER for Seq2Seq models"}
    )
    eval_wer_config: GenerationConfig = field(
        default_factory=lambda: GenerationConfig(),
        metadata={"help": "beam search config for evaluating wer during training"},
    )
    eval_wer_tokenizer: Any = field(
        default=None,
        metadata={"help": "tokenizer config for evaluating wer during training"},
    )
    eval_wer_post_process: str = field(
        default="letter",
        metadata={
            "help": "remove BPE tokens before scoring (can be sentencepiece, letter, and more)"
        },
    )
    autoregressive: bool = field(
        default=False,
        metadata={
            "help": "required for autoregressive decoders (like seq2seq models); "
            "adds 'prev_output_tokens' to input and appends eos to target"
        },
    )
    debug: bool = field(
        default=False,
    )

    gpt_path: str = field(
        default="",
        metadata={"help": "path of bart model"},
    )

    gpt_type: str = field(
        default="gpt2",
        metadata={"help": "path of bart model"},
    )

@register_task("asr_finetuning_prefix", dataclass=ASRFinetuningPrefixConfig)
class ASRFinetunig_Prefix(FairseqTask):
    def __init__(
        self,
        cfg: ASRFinetuningPrefixConfig,
    ):
        super().__init__(cfg)
        if cfg.eval_wer:
            assert cfg.labels is not None, "eval_wer can only be set during fine-tuning"
        self.blank_symbol = "<s>"
        print(cfg)
        print(cfg.gpt_path)
        
        from transformers import GPT2Tokenizer, GPT2Config
        self.gpt_tokenizer = GPT2Tokenizer.from_pretrained(pretrained_model_name_or_path = cfg.gpt_type, cache_dir = cfg.gpt_path)
        self.gpt_config = GPT2Config.from_pretrained(pretrained_model_name_or_path = cfg.gpt_type, cache_dir = cfg.gpt_path)
        dummpy_tgt_dict = self.load_target_dictionary(vocab_path = cfg.gpt_path)
        self.state.merge_state_dict({'target_dictionary': dummpy_tgt_dict})
        
    @classmethod
    def setup_task(cls, cfg: ASRFinetuningPrefixConfig, **kwargs):
        """Setup the task (e.g., load dictionaries).

        Args:
            cfg (ASRFinetuningConfig): configuration of this task
        """

        return cls(cfg)

    def load_target_dictionary(self, vocab_path):
        import json
        with open(vocab_path + 'vocab.json') as f:
            word2idx = json.load(f)
        idx2word = {v:k for k,v in word2idx.items()}
        tgt_dict = Dictionary()

        ## clear the pre-defined special symbols in fairseq
        tgt_dict.symbols = []
        tgt_dict.count = []
        tgt_dict.indices = {}
        ## add the words according to their idx
        for idx in range(len(idx2word)):
            tgt_dict.add_symbol(idx2word[idx])
        assert tgt_dict.indices == word2idx, "error when initializing word2index dict!"

        ## define the special tokens
        tgt_dict.bos_index = 1
        tgt_dict.pad_index = 2
        tgt_dict.unk_index = 3
        tgt_dict.eos_index = 50256
        return tgt_dict
    
    def encode(self, sentence):
        tokens = self.gpt_tokenizer.tokenize(sentence)
        # tokens = ' '.join(tokens.split()[:-1])
        # # if tokens[-1] == 220:
        #     # print(tokens)
        #     # tokens = tokens[:-1]

        ## only append [EOS]
        if len(tokens) > self.gpt_config.max_position_embeddings - 1:
            tokens = tokens[: self.gpt_config.max_position_embeddings - 1]
        bpe_sentence = tokens + [self.gpt_tokenizer.eos_token]
        return bpe_sentence
        
    def load_dataset(self, split: str, task_cfg: FairseqDataclass = None, **kwargs):
        data_path = self.cfg.data
        task_cfg = task_cfg or self.cfg

        # upgrade old task
        if isinstance(task_cfg, Namespace):
            if not hasattr(task_cfg, "autoregressive"):
                task_cfg.autoregressive = not task_cfg.criterion == 'ctc'

        manifest = os.path.join(data_path, "{}.tsv".format(split))
        self.datasets[split] = FileAudioDataset(
            manifest,
            sample_rate=task_cfg.get('sample_rate', self.cfg.sample_rate),
            max_sample_size=self.cfg.max_sample_size,
            min_sample_size=self.cfg.min_sample_size,
            pad=task_cfg.labels is not None or task_cfg.enable_padding,
            normalize=task_cfg.normalize,
        )

        if task_cfg.labels:
            label_path = os.path.join(data_path, f"{split}.{task_cfg.labels}")
            skipped_indices = getattr(self.datasets[split], 'skipped_indices', set())
            with open(label_path, "r") as f:
                labels = [ # encode using bpe
                    self.encode( # convert to lower case
                        ''.join(line.split()).lower().replace('|', ' ')
                    ) for i, line in enumerate(f)
                    if i not in skipped_indices
                ]

            assert len(labels) == len(self.datasets[split]), (
                    f"labels length ({len(labels)}) and dataset length "
                    f"({len(self.datasets[split])}) do not match")

            process_label = LabelEncoder(self.gpt_tokenizer)

            self.datasets[split] = AddTargetDataset(
                self.datasets[split],
                labels,
                pad=self.target_dictionary.pad(),
                eos=self.target_dictionary.eos(),
                batch_targets=True,
                process_label=process_label,
                add_to_input=task_cfg.get('autoregressive', False),
            )

    @property
    def source_dictionary(self):
        return None

    @property
    def target_dictionary(self):
        """Return the :class:`~fairseq.data.Dictionary` for the language
        model."""
        return self.state.target_dictionary

    def max_positions(self):
        """Maximum input length supported by the encoder."""
        return (sys.maxsize, sys.maxsize)

    def filter_indices_by_size(
        self,
        indices,
        dataset,
        max_positions=None,
        ignore_invalid_inputs=False,
    ):
        # we do not need to filter by size in this task as dataloaders take care of this
        return indices

    def valid_step(self, sample, model, criterion):
        loss, sample_size, logging_output = super().valid_step(sample, model, criterion)

        if self.cfg.eval_wer and self.cfg.autoregressive:
            metrics = self._inference_with_wer(self.sequence_generator, sample, model)
            logging_output["_num_char_errors"] = metrics["num_char_errors"]
            logging_output["_num_chars"] = metrics["num_chars"]
            logging_output["_num_word_errors"] = metrics["num_word_errors"]
            logging_output["_num_words"] = metrics["num_words"]
        return loss, sample_size, logging_output

    def build_model(self, model_cfg: FairseqDataclass):
        model = super().build_model(model_cfg)
        if self.cfg.eval_wer and self.cfg.autoregressive:
            self.sequence_generator = self.build_generator(
                [model],
                self.cfg.eval_wer_config,
            )
            if self.cfg.eval_wer_tokenizer:
                self.tokenizer = encoders.build_tokenizer(self.cfg.eval_wer_tokenizer)
            else:
                self.tokenizer = None
        return model

    def _inference_with_wer(self, generator, sample, model):
        import editdistance

        def decode(toks):
            s = self.gpt_tokenizer.decode(toks.detach().cpu().int().numpy())
            return s

        num_word_errors, num_char_errors = 0, 0
        num_chars, num_words = 0, 0
        gen_out = self.inference_step(generator, [model], sample, None)
        for i in range(len(gen_out)):
            hyp = decode(gen_out[i][0]["tokens"])
            ref = decode(
                utils.strip_pad(sample["target"][i], self.target_dictionary.pad()),
            )
            # print(gen_out[i][0]["tokens"])
            # print(sample["target"][i])
            print('test', hyp, ref)
            num_char_errors += editdistance.eval(hyp, ref)
            num_chars += len(ref)
            hyp_words = hyp.split()
            ref_words = ref.split()
            num_word_errors += editdistance.eval(hyp_words, ref_words)
            num_words += len(ref_words)

        return {
            "num_char_errors": num_char_errors,
            "num_chars": num_chars,
            "num_word_errors": num_word_errors,
            "num_words": num_words,
        }


    def reduce_metrics(self, logging_outputs, criterion):
        super().reduce_metrics(logging_outputs, criterion)

        zero = torch.scalar_tensor(0.0)
        num_char_errors = sum(
            log.get("_num_char_errors", zero) for log in logging_outputs
        )
        num_chars = sum(log.get("_num_chars", zero) for log in logging_outputs)
        num_word_errors = sum(
            log.get("_num_word_errors", zero) for log in logging_outputs
        )
        num_words = sum(log.get("_num_words", zero) for log in logging_outputs)
        metrics.log_scalar("_num_char_errors", num_char_errors)
        metrics.log_scalar("_num_chars", num_chars)
        metrics.log_scalar("_num_word_errors", num_word_errors)
        metrics.log_scalar("_num_words", num_words)
        if num_chars > 0:
            metrics.log_derived(
                "uer",
                lambda meters: meters["_num_char_errors"].sum
                * 100.0
                / meters["_num_chars"].sum
                if meters["_num_chars"].sum > 0
                else float("nan"),
            )
        if num_words > 0:
            metrics.log_derived(
                "wer",
                lambda meters: meters["_num_word_errors"].sum
                * 100.0
                / meters["_num_words"].sum
                if meters["_num_words"].sum > 0
                else float("nan"),
            )



