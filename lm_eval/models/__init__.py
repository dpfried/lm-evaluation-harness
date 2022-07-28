from . import gpt2
from . import gpt3
from . import dummy
from . import cm3
from . import incoder
from . import recoder_tagger

MODEL_REGISTRY = {
    "hf": gpt2.HFLM,
    "gpt2": gpt2.GPT2LM,
    "gpt3": gpt3.GPT3LM,
    "cm3": cm3.CM3LM,
    "incoder": incoder.InCoderLM,
    "dummy": dummy.DummyLM,
    "recoder_tagger": recoder_tagger.RecoderTagger,
}


def get_model(model_name):
    return MODEL_REGISTRY[model_name]
