# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os
import sys
from typing import Dict, List, Optional

import torch
from fairseq.models import (
    FairseqIncrementalDecoder,
    FairseqLanguageModel,
    register_model,
    register_model_architecture,
)


logger = logging.getLogger(__name__)


DEFAULT_MAX_TARGET_POSITIONS = 512


@register_model("hf_gpt2_v2")
class HuggingFaceGPT2LanguageModel(FairseqLanguageModel):
    def __init__(self, decoder):
        super().__init__(decoder)

    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        # fmt: off
        parser.add_argument('--embed-dim', type=int, metavar='N',
                            help='embedding dimension')
        parser.add_argument('--num-attention-heads', type=int, metavar='N',
                            help='num attention heads')
        parser.add_argument('--num-layers', type=int, metavar='N',
                            help='num layers')
        parser.add_argument('--dropout', type=float, metavar='D',
                            help='dropout probability for all fully connected layers '
                                 'in the embeddings, encoder, and pooler')
        parser.add_argument('--attention-dropout', type=float, metavar='D',
                            help='dropout probability for attention weights')
        # fmt: on

    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""
        default_architecture(args)
        return cls(HuggingFaceGPT2Decoder(args, task))


class HuggingFaceGPT2Decoder(FairseqIncrementalDecoder):
    def __init__(self, args, task):
        try:
            from transformers import GPT2Config, GPT2LMHeadModel
        except ImportError:
            raise ImportError(
                "\n\nPlease install huggingface/transformers with:"
                "\n\n  pip install transformers"
            )

        super().__init__(task.target_dictionary)

        config = GPT2Config(
            vocab_size=len(task.target_dictionary),
            n_positions=args.max_target_positions + 1,
            n_ctx=args.max_target_positions,
            n_embd=args.embed_dim,
            n_layer=args.num_layers,
            n_head=args.num_attention_heads,
            resid_pdrop=args.dropout,
            embd_pdrop=args.dropout,
            attn_pdrop=args.attention_dropout,
            layer_norm_epsilon=1e-6,
        )
        self.model = GPT2LMHeadModel(config)

        # set zero embedding for padding symbol
        self.pad_idx = task.target_dictionary.pad()
        self.model.transformer.wte.weight.data[self.pad_idx].zero_()
        self.model.transformer.wpe.weight.data[0].zero_()

    def forward(
        self,
        prev_output_tokens,
        src_lengths=None,
        incremental_state: Optional[Dict[str, List[torch.Tensor]]] = None,
        encoder_out=None,
    ):
        features = self.extract_features(prev_output_tokens, incremental_state)
        lm_logits = self.model.lm_head(features)
        return (lm_logits,)

    def extract_features(
        self,
        prev_output_tokens,
        incremental_state: Optional[Dict[str, List[torch.Tensor]]] = None,
    ):
        if incremental_state:
            past = self.get_incremental_state("past")
        else:
            past = None

        # don't attend to padding symbols
        attention_mask = prev_output_tokens.ne(self.pad_idx).int()

        # set position ids to exclude padding symbols
        position_ids = attention_mask * (
            torch.arange(1, 1 + prev_output_tokens.size(1))
            .to(prev_output_tokens)
            .repeat(prev_output_tokens.size(0), 1)
        )

        outputs = self.model.transformer(
            input_ids=prev_output_tokens,
            past=past,
            attention_mask=attention_mask,
            position_ids=position_ids,
        )
        last_hidden_states = outputs[0]

        if incremental_state:
            self.set_incremental_state(incremental_state, "past", outputs[1])

        return last_hidden_states

    def max_positions(self):
        return self.model.config.n_positions - 1


def base_lm_architecture(args):
    # backward compatibility for older model checkpoints
    if hasattr(args, "no_tie_adaptive_proj"):
        # previous models defined --no-tie-adaptive-proj, so use the existence of
        # that option to determine if this is an "old" model checkpoint
        args.no_decoder_final_norm = True  # old models always set this to True
        if args.no_tie_adaptive_proj is False:
            args.tie_adaptive_proj = True
    if hasattr(args, "decoder_final_norm"):
        args.no_decoder_final_norm = not args.decoder_final_norm

    args.dropout = getattr(args, "dropout", 0.1)
    args.attention_dropout = getattr(args, "attention_dropout", 0.0)

    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", 512)
    args.decoder_ffn_embed_dim = getattr(args, "decoder_ffn_embed_dim", 2048)
    args.decoder_layers = getattr(args, "decoder_layers", 6)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 8)
    args.adaptive_softmax_cutoff = getattr(args, "adaptive_softmax_cutoff", None)
    args.adaptive_softmax_dropout = getattr(args, "adaptive_softmax_dropout", 0)
    args.adaptive_softmax_factor = getattr(args, "adaptive_softmax_factor", 4)
    args.decoder_learned_pos = getattr(args, "decoder_learned_pos", False)
    args.activation_fn = getattr(args, "activation_fn", "relu")

    args.decoder_layerdrop = getattr(args, "decoder_layerdrop", 0)
    args.decoder_layers_to_keep = getattr(args, "decoder_layers_to_keep", None)
    args.quant_noise_pq = getattr(args, "quant_noise_pq", 0)
    args.quant_noise_pq_block_size = getattr(args, "quant_noise_pq_block_size", 8)
    args.quant_noise_scalar = getattr(args, "quant_noise_scalar", 0)

    args.base_layers = getattr(args, "base_layers", 0)
    args.base_sublayers = getattr(args, "base_sublayers", 1)
    args.base_shuffle = getattr(args, "base_shuffle", False)

    args.add_bos_token = getattr(args, "add_bos_token", False)
    args.no_token_positional_embeddings = getattr(
        args, "no_token_positional_embeddings", False
    )
    args.share_decoder_input_output_embed = getattr(
        args, "share_decoder_input_output_embed", False
    )
    args.character_embeddings = getattr(args, "character_embeddings", False)

    args.decoder_output_dim = getattr(
        args, "decoder_output_dim", args.decoder_embed_dim
    )
    args.decoder_input_dim = getattr(args, "decoder_input_dim", args.decoder_embed_dim)

    # Model training is not stable without this
    args.decoder_normalize_before = True
    args.no_decoder_final_norm = getattr(args, "no_decoder_final_norm", False)

    args.adaptive_input = getattr(args, "adaptive_input", False)
    args.adaptive_input_factor = getattr(args, "adaptive_input_factor", 4)
    args.adaptive_input_cutoff = getattr(args, "adaptive_input_cutoff", None)

    args.tie_adaptive_weights = getattr(args, "tie_adaptive_weights", False)
    args.tie_adaptive_proj = getattr(args, "tie_adaptive_proj", False)

    args.no_scale_embedding = getattr(args, "no_scale_embedding", False)
    args.layernorm_embedding = getattr(args, "layernorm_embedding", False)
    args.checkpoint_activations = getattr(args, "checkpoint_activations", False)
    args.offload_activations = getattr(args, "offload_activations", False)
    if args.offload_activations:
        args.checkpoint_activations = True


# @register_model_architecture("transformer_lm", "transformer_lm_gpt")
def transformer_lm_gpt(args):
    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", 768)
    args.decoder_ffn_embed_dim = getattr(args, "decoder_ffn_embed_dim", 3072)
    args.decoder_layers = getattr(args, "decoder_layers", 12)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 12)
    args.dropout = getattr(args, "dropout", 0.1)
    args.attention_dropout = getattr(args, "attention_dropout", 0.1)
    args.activation_fn = getattr(args, "activation_fn", "gelu")
    base_lm_architecture(args)


