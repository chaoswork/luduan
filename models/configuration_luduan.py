from transformers.configuration_utils import PretrainedConfig
from transformers.utils import logging


logger = logging.get_logger(__name__)


class LuduanConfig(PretrainedConfig):
    model_type = "luduan"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
            self,
            vocab_size=32000,
            n_embd=4096,
            intermediate_size=11008,
            n_layer=32,
            n_head=32,
            block_size=2048,
            dropout=0.0,
            hidden_act="silu",
            rms_norm_eps=1e-6,
            pad_token_id=0,
            bos_token_id=1,
            eos_token_id=2,
            tie_word_embeddings=False,
            **kwargs,
        ):
        self.vocab_size = vocab_size
        self.block_size = block_size
        self.n_embd = n_embd
        self.intermediate_size = intermediate_size
        self.rms_norm_eps = rms_norm_eps
        self.n_layer = n_layer
        self.n_head = n_head
        self.dropout = dropout
        self.hidden_act = hidden_act
#        self.use_cache = use_cache
        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )
