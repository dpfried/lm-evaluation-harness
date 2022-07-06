import os
import gzip
import json
from typing import List, Tuple, Dict
from collections import namedtuple
import re

from functools import lru_cache

import numpy as np
import torch

from lm_eval.base import rf, PerplexityTask
from lm_eval.metrics import (
    mean, perplexity, weighted_perplexity, weighted_mean, token_count, precision_from_sufficient_stats, recall_from_sufficient_stats, f1_score_from_sufficient_stats, acc_from_sufficient_stats
)

OPERATION_NAMES = "delete edit insert".split()
# these will be renormalized to give the probability of each operation
NoisingWeights = namedtuple("NoisingWeights", OPERATION_NAMES)
UNIFORM_NOISING_WEIGHTS = NoisingWeights(1.0, 1.0, 1.0)

NoiseOperation = namedtuple("NoiseOperation", [
    # Tuple[int, int]
    "original_span",
    # int, corresponding to CausallyEditedDataset.{DELETE, EDIT, INSERT}
    "operation",
    # List[int]
    "noised_tokens",
    # bool
    "completed",
])

NoisedDocument = namedtuple("NoisedDocument", [
    # List[int]
    "original_document", 
    # List[NoiseOperation]
    "noise_operations",
    # int
    "retries",
    # bool
    "completed",
])

def _nd_to_jsonable(noised_document: NoisedDocument) -> dict:
    d = noised_document._replace(noise_operations=[op._asdict() for op in noised_document.noise_operations])
    return d._asdict()

def _no_from_jsonable(dictionary: dict) -> NoiseOperation:
    to_del = [key for key in dictionary if key not in NoiseOperation._fields]
    for key in to_del:
        del dictionary[key]
    return NoiseOperation(**dictionary)

def _nd_from_jsonable(dictionary: dict) -> NoisedDocument:
    # warning: destructive
    if 'original_ids' in dictionary:
        dictionary['original_document'] = dictionary['original_ids']
    dictionary["noise_operations"] = [_no_from_jsonable(d) for d in dictionary["noise_operations"]]

    to_del = [key for key in dictionary if key not in NoisedDocument._fields]
    for key in to_del:
        del dictionary[key]
    return NoisedDocument(**dictionary)

NoisedDocument.to_jsonable = _nd_to_jsonable
NoisedDocument.from_jsonable = _nd_from_jsonable
NoiseOperation.from_jsonable = _no_from_jsonable

def span_intersection(left: Tuple[int, int], right: Tuple[int, int]) -> bool:
    left_x, left_y = left
    right_x, right_y = right
    return max(left_x, right_x) < min(left_y, right_y)

def span_or_location_intersection(left: Tuple[int, int], right: Tuple[int, int]) -> bool:
    left_x, left_y = left
    right_x, right_y = right
    if left_x == left_y:
        # this is a location; check for containment within right.
        left_loc = left_x
        # we use < < rather than <= < because it's ok to insert before a span the beginning of a span
        return right_x < left_loc < right_y
    if right_x == right_y:
        # this is a location; check for containment within left.
        right_loc = right_x
        # we use < < rather than <= < because it's ok to insert before a span the beginning of a span
        return left_x < right_loc < left_y
    return max(left_x, right_x) < min(left_y, right_y)

class Recoder(PerplexityTask):
    """if we constrained to only DELETE operations, we should recover CausallyMaskedDataset"""
    VERSION = 0
    LANG_NAME = None
    ROOT_DIR = None

    VERBOSE = True


    def fewshot_description(self):
        # TODO: figure out fewshot description
        return ""

    def has_validation_docs(self):
        return True

    def has_train_docs(self):
        return False

    def has_test_docs(self):
        return False

    def validation_docs(self):
        yield from self.read_data(self.FILENAME)
                
    def train_docs(self):
        pass

    def test_docs(self):
        pass

    def construct_requests(self, doc, ctx):
        assert not ctx
        noised_part, all_end_parts = self.build_parts(doc)
        full_doc = self.apply_noise_to_parts(noised_part, all_end_parts)
        full_str = self.tokenizer.decode(full_doc, clean_up_tokenization_spaces=False)
        if "<|endofnoised|>" in full_str:
            noised_str = full_str[:full_str.index("<|endofnoised|>")] + "<|endofnoised|>"
        else:
            noised_str = full_str
        req = [
            rf.loglikelihood_rolling(noised_str),
            rf.loglikelihood_rolling(full_str),
            rf.greedy_until(noised_str, ["<|endoftext|>"])
        ]
        return req

    def process_results(self, doc: NoisedDocument, results):
        (noised_loglikelihood, noised_num_tokens),\
        (full_loglikelihood, full_num_tokens),\
            pred = results
        noised_part, all_end_parts = self.build_parts(doc)
        gold = self.tokenizer.decode([t for et in all_end_parts for t in et], clean_up_tokenization_spaces=False)
        
        gold_labels, gold_spans = self.unnoised_to_labels(gold)
        pred_labels, pred_spans = self.unnoised_to_labels(pred)
        if self.VERBOSE:
            print("-"*40)
            print("original:\t", )
            og = self.tokenizer.decode(noised_part, clean_up_tokenization_spaces=False)
            print(og)
            print("gold:\t", gold)
            print("pred:\t", pred)
            print("gold edits:")
            for ix in [0,1,2]:
                print(f"{ix}: ", (gold_labels == ix).nonzero().flatten())
            for start, end, label in gold_spans:
                print(label, self.tokenizer.decode(noised_part[start:end], clean_up_tokenization_spaces=False))
            print("pred edits:")
            for ix in [0,1,2]:
                print(f"{ix}: ", (pred_labels == ix).nonzero().flatten())
            for start, end, label in pred_spans:
                print(label, self.tokenizer.decode(noised_part[start:end], clean_up_tokenization_spaces=False))

        gold_unlabels = (gold_labels != self.KEEP)
        pred_unlabels = (pred_labels != self.KEEP)
        
        # if self.VERBOSE:
        #     labeled_acc = (gold_labels == pred_labels).sum().float() / gold_labels.size(0)
        #     unlabeled_acc = (gold_unlabels == pred_unlabels).sum().float() / gold_unlabels.size(0)
        #     print(f"labeled_acc: {labeled_acc:.2f}")
        #     print(f"unlabeled_acc: {unlabeled_acc:.2f}")
        #     print()
        unnoised_loglikelihood = full_loglikelihood - noised_loglikelihood
        unnoised_num_tokens = full_num_tokens - noised_num_tokens

        unlabeled_acc_ss = self.compute_unlabeled_acc_sufficient_stats(gold_unlabels, pred_unlabels)
        unlabeled_f1_ss = self.compute_unlabeled_f1_sufficient_stats(gold_unlabels, pred_unlabels)
        edit_f1_ss = self.compute_unlabeled_f1_sufficient_stats(gold_labels == self.EDIT, pred_labels == self.EDIT)
        delete_f1_ss = self.compute_unlabeled_f1_sufficient_stats(gold_labels == self.DELETE, pred_labels == self.DELETE)
        if self.VERBOSE:
            print("edit_f1_ss:", edit_f1_ss)
            print("delete_f1_ss:", delete_f1_ss)
            print("unlabeled_f1_ss:", unlabeled_f1_ss)
        return {
            "noised_perplexity": (noised_loglikelihood, noised_num_tokens),
            "unnoised_perplexity": (unnoised_loglikelihood, unnoised_num_tokens),
            "full_perplexity": (full_loglikelihood, full_num_tokens),
            "num_noised_model_tokens": noised_num_tokens,
            "num_unnoised_model_tokens": full_num_tokens - noised_num_tokens,
            "num_full_model_tokens": full_num_tokens,
            "unlabeled_acc": unlabeled_acc_ss,
            "unlabeled_precision": unlabeled_f1_ss,
            "edit_precision": edit_f1_ss,
            "delete_precision": delete_f1_ss,
            "unlabeled_recall": unlabeled_f1_ss,
            "edit_recall": edit_f1_ss,
            "delete_recall": delete_f1_ss,
            "unlabeled_f1": unlabeled_f1_ss,
            "edit_f1": edit_f1_ss,
            "delete_f1": delete_f1_ss,
        }

    def aggregation(self):
        return {
            "noised_perplexity": weighted_perplexity,
            "unnoised_perplexity": weighted_perplexity,
            "full_perplexity": weighted_perplexity,
            "num_noised_model_tokens": token_count,
            "num_unnoised_model_tokens": token_count,
            "num_full_model_tokens": token_count,
            "unlabeled_acc": acc_from_sufficient_stats,
            "unlabeled_precision": precision_from_sufficient_stats,
            "edit_precision": precision_from_sufficient_stats,
            "delete_precision": precision_from_sufficient_stats,
            "unlabeled_recall": recall_from_sufficient_stats,
            "edit_recall": recall_from_sufficient_stats,
            "delete_recall": recall_from_sufficient_stats,
            "unlabeled_f1": f1_score_from_sufficient_stats,
            "edit_f1": f1_score_from_sufficient_stats,
            "delete_f1": f1_score_from_sufficient_stats,
        }

    @staticmethod
    def compute_unlabeled_f1_sufficient_stats(gold, pred):
        true_positives = (gold & pred).sum()
        total_gold = gold.sum()
        total_pred = pred.sum()

        return true_positives, total_gold, total_pred

    @staticmethod
    def compute_unlabeled_acc_sufficient_stats(gold, pred):
        return (gold == pred).sum(), gold.size(0)

    def unnoised_to_labels(self, unnoised_str):
        tokens = self.tokenizer.tokenize(unnoised_str)
        labels = torch.full((self.tokens_per_sample,), self.KEEP)
        spans = []
        for i in range(len(tokens)):
            if tokens[i] in self.TOKEN_TO_LABEL:
                label = self.TOKEN_TO_LABEL[tokens[i]]
                # TODO: compute accuracy for INSERT operations
                if label == self.INSERT_TOKEN:
                    continue
                if i+3 > len(tokens):
                    continue
                start, end = tokens[i+1:i+3]
                if not (start.startswith("<|pos") and end.startswith("<|pos")):
                    continue
                start = self.parse_position(start)
                end = self.parse_position(end)
                spans.append((start, end+1, label))
                if start is None or end is None:
                    continue
                labels[start:end+1] = label
        return labels, spans
    
    FILENAME = None

    KEEP = -1
    DELETE = 0
    EDIT = 1
    INSERT = 2

    NOISED_EOS_TOKEN = "<|endofnoised|>"
    DELETE_TOKEN = "<|delete|>"
    EDIT_TOKEN = "<|edit|>"
    INSERT_TOKEN = "<|insert|>"

    TOKEN_TO_LABEL = {
        DELETE_TOKEN: DELETE,
        EDIT_TOKEN: EDIT,
        INSERT_TOKEN: INSERT,
    }

    @staticmethod
    def make_position(i: int):
        return f"<|pos:{i}|>"

    POSITION_REGEX = re.compile(r"<\|pos:(\d+)\|>")

    @staticmethod
    def parse_position(token: str):
        match = Recoder.POSITION_REGEX.match(token)
        if not match:
            return None
        return int(match.groups()[0])

    @staticmethod
    def exists(path):
        if not path.endswith(".jsonl"):
            path = path + ".jsonl"
        return os.path.exists(path)

    def __init__(self):
        super().__init__()
        tokens_per_sample = 2048
        tokenizer, noised_eos, sentinel_eos, sentinel_token_ids_per_operation, position_token_ids = make_tokenizer(tokens_per_sample)
        self.tokenizer = tokenizer
        self.sentinel_token_ids_per_operation = sentinel_token_ids_per_operation
        self.position_token_ids = position_token_ids
        self.tokens_per_sample = tokens_per_sample
        self.noised_eos = noised_eos
        self.sentinel_eos = sentinel_eos
        assert self.tokens_per_sample + 1 == len(self.position_token_ids)

        # self.read_data(filename, show_progress)

    def read_data(self, filename, show_progress:bool = False):
        # noised_docs = []
        with open(filename, "r", encoding="utf-8") as f:
            it = f
            if show_progress:
                import tqdm
                it = tqdm.tqdm(it, ncols=80)
            for line in it:
                data = json.loads(line.strip())
                noise_doc = NoisedDocument.from_jsonable(data)
                yield noise_doc
                # noise_applied: List[int] = self.apply_noise(noise_doc)
                # noised_docs.append(noise_doc)
        # self.noised_docs = noised_docs

    def get_operation_sentinel(self, operation: int) -> int:
        return self.sentinel_token_ids_per_operation[operation]

    def get_position_sentinel(self, position: int) -> int:
        return self.position_token_ids[position]

    def build_parts(self, noise_document: NoisedDocument):
        # define a new KEEP operation, indexed by -1, which is all spans which
        # aren't otherwise included in an edit operation
        KEEP = -1
        operations: List[NoiseOperation] = noise_document.noise_operations
        keep_operations: List[NoiseOperation] = []
        last_end = 0
        document = noise_document.original_document
        for op in sorted(operations, key=lambda op: op.original_span):
            # need the int cast here because np.int64 is not json serializable
            start, end = tuple(map(int, op.original_span))
            keep_operations.append(NoiseOperation((last_end, start), KEEP, noise_document.original_document[last_end:start], True))
            last_end = end
        keep_operations.append(NoiseOperation((last_end, len(document)), KEEP, noise_document.original_document[last_end:len(document)], True))
        if len(operations) == 0:
            assert len(keep_operations) == 1
            assert keep_operations[0].original_span == (0, len(document))
        elif len(operations) == 1:
            assert len(keep_operations) == 2
            assert keep_operations[0].original_span == (0, operations[0].original_span[0])
            assert keep_operations[1].original_span == (operations[0].original_span[1], len(document))

        # TODO: consider allowing non-sorted
        all_operations = list(sorted(operations + keep_operations, key=lambda op: tuple(op.original_span)))
        # print(all_operations)

        # ensure all tokens in the original are included
        assert sum(op.original_span[1] - op.original_span[0] for op in all_operations) == len(document)

        assert len(all_operations) > 0

        noised_document = []
        all_end_tokens = []

        for operation in all_operations:
            start, end = operation.original_span
            original_tokens = document[start:end]

            noised_start = len(noised_document)
            noised_end = len(noised_document) + len(operation.noised_tokens) - 1

            end_tokens = []

            if operation.operation == KEEP:
                noised_document.extend(original_tokens)
                continue
            if noised_start >= self.tokens_per_sample or noised_end >= self.tokens_per_sample:
                # we won't have positional embeddings for these
                break
            if operation.operation == self.DELETE:
                # no noised_tokens to insert
                assert len(operation.noised_tokens) == 0
                # INSERT is the opposite of the DELETE noising operation
                end_tokens.append(self.get_operation_sentinel(self.INSERT))
                end_tokens.append(self.get_position_sentinel(noised_start))
                end_tokens.extend(original_tokens)
                end_tokens.append(self.sentinel_eos)
            elif operation.operation == self.EDIT:
                noised_document.extend(operation.noised_tokens)
                # EDIT is its own opposite
                end_tokens.append(self.get_operation_sentinel(self.EDIT))
                end_tokens.append(self.get_position_sentinel(noised_start))
                end_tokens.append(self.get_position_sentinel(noised_end))
                end_tokens.extend(original_tokens)
                end_tokens.append(self.sentinel_eos)
            elif operation.operation == self.INSERT:
                noised_document.extend(operation.noised_tokens)
                # DELETE is the opposite of the INSERT noising operation
                end_tokens.append(self.get_operation_sentinel(self.DELETE))
                end_tokens.append(self.get_position_sentinel(noised_start))
                end_tokens.append(self.get_position_sentinel(noised_end))
                end_tokens.extend(original_tokens)
                end_tokens.append(self.sentinel_eos)
            all_end_tokens.append(end_tokens)
        return noised_document, all_end_tokens

    def apply_noise_to_parts(self, noised_part, all_end_parts):
        end_parts = [t for et in all_end_parts for t in et]
        return (noised_part + [self.noised_eos] + end_parts)[:self.tokens_per_sample]

    def apply_noise(self, noise_document: NoisedDocument):
        noised_part, all_end_parts = self.build_parts(noise_document)
        return self.apply_noise_to_parts(noised_part, all_end_parts)

class RecoderPython512(Recoder):
    FILENAME = "/checkpoint/dpf/data/code_corpus_noised/512-512_mr-2_mdr-2/val_github_python_forkless_open-source_2star+_0.jsonl"

def make_tokenizer(tokens_per_sample):
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("facebook/incoder-1B")

    tokenizer.add_tokens(Recoder.NOISED_EOS_TOKEN)
    noised_eos = tokenizer.vocab[Recoder.NOISED_EOS_TOKEN]

    sentinel_token_ids_per_operation = {}
    for operation, token_str in [
        (Recoder.DELETE, Recoder.DELETE_TOKEN),
        (Recoder.EDIT, Recoder.EDIT_TOKEN),
        (Recoder.INSERT, Recoder.INSERT_TOKEN),
    ]:
        tokenizer.add_tokens(token_str, special_tokens=True)
        sentinel_token_ids_per_operation[operation] = tokenizer.vocab[token_str]

    sentinel_eos = tokenizer.vocab["<|endofmask|>"]

    position_token_ids = []
    for i in range(tokens_per_sample+1):
        token_str = Recoder.make_position(i)
        tokenizer.add_tokens(token_str, special_tokens=True)
        position_token_ids.append(tokenizer.vocab[token_str])
    return tokenizer, noised_eos, sentinel_eos, sentinel_token_ids_per_operation, position_token_ids