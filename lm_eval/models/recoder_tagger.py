from lm_eval import utils

from tqdm import tqdm
import torch
import transformers
import tokenizers

import contextlib

from transformers import AutoModel, PreTrainedModel, BartConfig
from transformers.modeling_outputs import TokenClassifierOutput
from transformers.models.xglm import XGLMConfig
# from transformers.models.t5.modeling_t5 import T5Stack
from transformers.models.bart.modeling_bart import BartEncoder

from typing import Optional, Union, Tuple, List

import torch
from torch import nn
from torch.nn import CrossEntropyLoss

from lm_eval.base import CacheHook

class TaggerConfig(XGLMConfig):
    def __init__(self,
                 num_labels: int = 4,
                 classifier_dropout: float = 0.1,
                 freeze_encoder: bool = False,
                 # number of layers of bidirectional attention to apply on top of the XGLM model
                 num_post_layers: int = 0,
                 encoder_output_layer: List[int] = -1,
                 loss="cross_entropy",
                 loss_weights=None,
                 **kwargs):
        super().__init__(**kwargs)
        self.num_labels = num_labels
        self.classifier_dropout = classifier_dropout
        self.freeze_encoder = freeze_encoder

        self.num_post_layers = num_post_layers
        self.encoder_output_layer = encoder_output_layer

        self.loss = loss
        self.loss_weights = loss_weights
        print(f"freeze encoder: {freeze_encoder}")

class Tagger(PreTrainedModel):

    config_class = TaggerConfig
    supports_gradient_checkpointing = True

    def __init__(self, config: TaggerConfig, loss_weights = None):
        super().__init__(config)
        self.config = config
        self.num_labels = config.num_labels
        assert self.num_labels is not None
        self.model = AutoModel.from_pretrained(config._name_or_path)
        self.dropout = nn.Dropout(config.classifier_dropout)
        self.linear = nn.Linear(config.d_model, self.num_labels)

        if loss_weights is not None:
            loss_weights = nn.Parameter(torch.tensor(loss_weights), requires_grad=False)
        self.loss_weights = loss_weights
        # self.post_init()

        if self.config.num_post_layers > 0:
            post_config = BartConfig(
                encoder_layers=self.config.num_post_layers,
                max_position_embeddings=2048,
                encoder_ffn_dim=self.config.ffn_dim,
                encoder_attention_heads=self.config.attention_heads,
                encoder_layerdrop=self.config.layerdrop,
                activation_function=self.config.activation_function,
                d_model=self.config.d_model,
                dropout=self.config.dropout,
                attention_dropout=self.config.attention_dropout,
                activation_dropout=self.config.activation_dropout,
                is_encoder_decoder=False,
                )
            self.post_encoder = BartEncoder(post_config)
        else:
            self.post_encoder = None

        # TODO: call self.post_init() ?

    def _init_weights(self, module):
        raise NotImplementedError()

    def _set_gradient_checkpointing(self, module, value=False):
        module.gradient_checkpointing = value

    def _compute_logits(self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
    ):
        with torch.no_grad() if self.config.freeze_encoder else contextlib.nullcontext():
            use_hidden = hasattr(self.config, 'encoder_output_layer') and self.config.encoder_output_layer != [-1]
            encoder_outputs = self.model(
                input_ids,
                attention_mask=attention_mask,
                # token_type_ids=token_type_ids,
                # position_ids=position_ids,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                output_attentions=False,
                output_hidden_states=use_hidden,
                return_dict=True,
            )
            if use_hidden:
                hs = []
                for l in self.config.encoder_output_layer:
                    hs.append(encoder_outputs.hidden_states[l])
                if len(hs) == 1:
                    sequence_output = hs[0]
                else:
                    sequence_output = torch.stack(hs, 0).mean(0)
            else:
                sequence_output = encoder_outputs[0]
            sequence_output = self.dropout(sequence_output)

        if self.post_encoder is not None:
            post_encoder_outputs = self.post_encoder(
                inputs_embeds=sequence_output, 
                attention_mask=attention_mask,
            )
            sequence_output = post_encoder_outputs[0]

        logits = self.linear(sequence_output)
        return logits

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], TokenClassifierOutput]:
        logits = self._compute_logits(input_ids, attention_mask, head_mask, inputs_embeds)
        loss = None
        if labels is not None:
            assert (labels[:,0] >= 0).all()
            if self.config.loss == "cross_entropy":
                loss_fct = CrossEntropyLoss(self.loss_weights)
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.loss == "max_margin":
                raise NotImplementedError("loss weights for loss=max_margin")
            else:
                raise NotImplementedError(self.config.loss)

        if not return_dict:
            output = (logits,)
            # output = (logits,) + encoder_outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            # hidden_states=hidden_states,
            # attentions=encoder_outputs.attentions,
            # TODO: maybe return both encoder and post_encoder attentions and hidden_states, but it seems to slow down compute_metrics
            hidden_states=None,
            attentions=None,
        )

class RecoderTagger():
    @classmethod
    def create_from_arg_string(cls, arg_string, additional_config=None):
        additional_config = {} if additional_config is None else additional_config
        args = utils.simple_parse_args_string(arg_string)
        args2 = {k: v for k, v in additional_config.items() if v is not None}
        return cls(**args, **args2)

    def set_cache_hook(self, cache_hook):
        self.cache_hook = cache_hook

    def __init__(
        self,
        device="cuda",
        pretrained="gpt2",
        subfolder=None,
        tokenizer=None,
        batch_size=1,
    ):
        super().__init__()

        assert isinstance(device, str)
        assert isinstance(pretrained, str)
        assert isinstance(batch_size, int)

        if device:
            if device not in ["cuda", "cpu"]:
                device = int(device)
            self._device = torch.device(device)
            print(f"Using device '{device}'")
        else:
            print("Device not specified")
            print(f"Cuda Available? {torch.cuda.is_available()}")
            self._device = (
                torch.device("cuda")
                if torch.cuda.is_available()
                else torch.device("cpu")
            )

        # TODO: update this to be less of a hack once subfolder is fixed in HF
        self.model = Tagger.from_pretrained(
            pretrained,
        ).to(self._device)
        self.model.eval()

        tokenizer_name = pretrained if tokenizer is None else tokenizer
        # pretrained tokenizer for neo is broken for now so just hard-coding this to gpt2
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            tokenizer_name,
            # revision=revision,
            subfolder=subfolder,
        )

        self.tokenizer = tokenizer

        # multithreading and batching
        self.batch_size_per_gpu = batch_size 

        # TODO: fix multi-gpu
        # gpus = torch.cuda.device_count()
        # if gpus > 1:
        #     self.gpt2 = nn.DataParallel(self.gpt2)
        self.cache_hook = CacheHook(None)

    def set_cache_hook(self, cache_hook):
        self.cache_hook = cache_hook

    def _model_tag(self, tokens):
        assert isinstance(tokens, list)
        tokens = torch.tensor(tokens + [self.tokenizer.bos_token_id])[:2048]
        tokens = tokens.to(self.model.device)
        tokens = tokens.unsqueeze(0)
        with torch.no_grad():
            outputs = self.model(input_ids=tokens, return_dict=True)
        predictions = (outputs.logits.argmax(-1) - 1).flatten()
        return predictions.cpu().tolist()

    @property
    def device(self):
        return self._device

    def tag_tokens(self, requests):
        res = []

        # def _collate(x):
        #     toks = self.tok_encode(x[0])
        #     return len(toks), x[0]

        # re_ord = utils.Reorderer(requests, _collate)

        # for tokens in tqdm(re_ord.get_reordered()):
        for tokens, in tqdm(requests):
            tags = self._model_tag(tokens)
            # partial caching
            self.cache_hook.add_partial("tag_tokens", tokens, tags)
            res.append(tags)

        # return re_ord.get_original(res)
        return res