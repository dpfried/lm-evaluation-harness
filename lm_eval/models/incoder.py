from .gpt2 import HFLM

class InCoderLM(HFLM):
    def check_tokenizer(self, tokenizer, tokenizer_name):
        pass
        # assert tokenizer_name.startswith("facebook/incoder")

    def tok_encode(self, string: str):
        return self.tokenizer.encode(string)

    def tok_decode(self, tokens):
        return self.tokenizer.decode(tokens, clean_up_tokenization_spaces=False)

    @property
    def eot_token_id(self):
        # TODO: tok_encode will also prepend <|endoftext|>, so this results in
        # being forced to generate it an extra time. But doing it correctly would need
        # modifications either to base.py, or loglikelihood & loglikelihood_rolling
        return self.tokenizer.encode("<|endoftext|>", add_special_tokens=False)[0]
