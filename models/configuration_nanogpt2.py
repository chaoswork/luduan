from transformers.configuration_utils import PretrainedConfig
from transformers.utils import logging


logger = logging.get_logger(__name__)


class NanoGPT2Config(PretrainedConfig):
    model_type = "NanoGPT2"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
            self,
            # vocab_size=64000,
            vocab_size=50257,
            n_embd=768,
            n_layer=12,
            n_head=12,
            block_size=1024,
            dropout=0.0,
            bias=True,
            pad_token_id=0,
            bos_token_id=1,
            eos_token_id=2,
            tie_word_embeddings=False,
            **kwargs,
        ):
        self.vocab_size = vocab_size
        self.block_size = block_size
        self.n_embd = n_embd
        self.n_layer = n_layer
        self.n_head = n_head
        self.dropout = dropout
        self.bias = bias
#        self.use_cache = use_cache
        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )
