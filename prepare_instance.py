
from pytorch_pretrained_bert import BertTokenizer
from tqdm import tqdm, trange
import json
from tempfile import TemporaryDirectory
from pathlib import Path
import shelve
import numpy as np
from random import random, randint, shuffle, choice, sample
from my_utils import makedir_and_clear
import os
import logging

class DocumentDatabase:
    def __init__(self, reduce_memory=False):
        if reduce_memory:
            self.temp_dir = TemporaryDirectory()
            self.working_dir = Path(self.temp_dir.name)
            self.document_shelf_filepath = self.working_dir / 'shelf.db'
            self.document_shelf = shelve.open(str(self.document_shelf_filepath),
                                              flag='n', protocol=-1)
            self.documents = None
        else:
            self.documents = []
            self.document_shelf = None
            self.document_shelf_filepath = None
            self.temp_dir = None
        self.doc_lengths = []
        self.doc_cumsum = None
        self.cumsum_max = None
        self.reduce_memory = reduce_memory

    def add_document(self, document):
        if self.reduce_memory:
            current_idx = len(self.doc_lengths)
            self.document_shelf[str(current_idx)] = document
        else:
            self.documents.append(document)
        self.doc_lengths.append(len(document))

    def _precalculate_doc_weights(self):
        self.doc_cumsum = np.cumsum(self.doc_lengths)
        self.cumsum_max = self.doc_cumsum[-1]

    def sample_doc(self, doc_num, current_idx, sentence_weighted=True):

        # Uses the current iteration counter to ensure we don't sample the same doc twice
        if sentence_weighted:
            # With sentence weighting, we sample docs proportionally to their sentence length
            if self.doc_cumsum is None or len(self.doc_cumsum) != len(self.doc_lengths):
                self._precalculate_doc_weights()
            rand_start = self.doc_cumsum[current_idx]
            rand_end = rand_start + self.cumsum_max - self.doc_lengths[current_idx]
            sentence_index = randint(rand_start, rand_end) % self.cumsum_max
            sampled_doc_index = np.searchsorted(self.doc_cumsum, sentence_index, side='right')
        else:
            # If we don't use sentence weighting, then every doc has an equal chance to be chosen
            sampled_doc_index = current_idx + randint(1, len(self.doc_lengths)-1)

        if sampled_doc_index == current_idx:
            logging.debug("sample_doc sampled_doc_index == current_idx = {}".format(current_idx))
            sampled_doc_index = (current_idx+1) % doc_num

        if self.reduce_memory:
            return self.document_shelf[str(sampled_doc_index)]
        else:
            return self.documents[sampled_doc_index]

    def __len__(self):
        return len(self.doc_lengths)

    def __getitem__(self, item):
        if self.reduce_memory:
            return self.document_shelf[str(item)]
        else:
            return self.documents[item]

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, traceback):
        if self.document_shelf is not None:
            self.document_shelf.close()
        if self.temp_dir is not None:
            self.temp_dir.cleanup()

def truncate_seq_pair(tokens_a, tokens_b, max_num_tokens, tokens_a_ent, tokens_b_ent,
                      tokens_a_wp_to_ent, tokens_b_wp_to_ent):
    """Truncates a pair of sequences to a maximum sequence length. Lifted from Google's BERT repo."""
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_num_tokens:
            break

        trunc_a = True if len(tokens_a) > len(tokens_b) else False
        if trunc_a:
            trunc_tokens = tokens_a
            trunc_tokens_ent = tokens_a_ent
            trunc_tokens_wp_to_ent = tokens_a_wp_to_ent
        else:
            trunc_tokens = tokens_b
            trunc_tokens_ent = tokens_b_ent
            trunc_tokens_wp_to_ent = tokens_b_wp_to_ent

        # trunc_tokens = tokens_a if len(tokens_a) > len(tokens_b) else tokens_b
        # assert len(trunc_tokens) >= 1
        # feili
        if len(trunc_tokens) < 1:
            logging.debug("truncate_seq_pair: len(trunc_tokens) < 1: tokens_a {} tokens_b {}".format(tokens_a,tokens_b))
            trunc_tokens.append('[UNK]')
            trunc_tokens_ent.append(0)
            trunc_tokens_wp_to_ent.append(-1)

        # We want to sometimes truncate from the front and sometimes from the
        # back to add more randomness and avoid biases.
        if random() < 0.5:
            del trunc_tokens[0]
            del trunc_tokens_ent[0]
            del trunc_tokens_wp_to_ent[0]
        else:
            trunc_tokens.pop()
            trunc_tokens_ent.pop()
            trunc_tokens_wp_to_ent.pop()


def create_masked_lm_predictions(tokens, masked_lm_prob, max_predictions_per_seq, vocab_list):
    """Creates the predictions for the masked LM objective. This is mostly copied from the Google BERT repo, but
    with several refactors to clean it up and remove a lot of unnecessary variables."""
    cand_indices = []
    for (i, token) in enumerate(tokens):
        if token == "[CLS]" or token == "[SEP]":
            continue
        cand_indices.append(i)

    num_to_mask = min(max_predictions_per_seq,
                      max(1, int(round(len(tokens) * masked_lm_prob))))
    shuffle(cand_indices)
    mask_indices = sorted(sample(cand_indices, num_to_mask))
    masked_token_labels = []
    for index in mask_indices:
        # 80% of the time, replace with [MASK]
        if random() < 0.8:
            masked_token = "[MASK]"
        else:
            # 10% of the time, keep original
            if random() < 0.5:
                masked_token = tokens[index]
            # 10% of the time, replace with random word
            else:
                masked_token = choice(vocab_list)
        masked_token_labels.append(tokens[index])
        # Once we've saved the true label for that token, we can overwrite it with the masked version
        tokens[index] = masked_token

    return tokens, mask_indices, masked_token_labels

import copy
def create_masked_lm_and_ner_predictions(tokens, masked_lm_prob, max_predictions_per_seq, vocab_list, tokens_ent_mask):

    tokens = copy.deepcopy(tokens)
    cand_indices_ent = []

    # sample entity token as much as possible
    for (i, token) in enumerate(tokens):
        if tokens_ent_mask[i] == 1 and token != "[CLS]" and token != "[SEP]":
            cand_indices_ent.append(i)

    # get the num of masked token
    num_to_mask = min(max_predictions_per_seq,
                      max(1, int(round(len(tokens) * masked_lm_prob)))) - len(cand_indices_ent)
    if num_to_mask < 1:
        num_to_mask = 1

    cand_indices = []
    for (i, token) in enumerate(tokens):
        if token == "[CLS]" or token == "[SEP]":
            continue
        if i in cand_indices_ent:
            continue
        cand_indices.append(i)

    shuffle(cand_indices)
    mask_indices = sample(cand_indices, num_to_mask)

    mask_indices.extend(cand_indices_ent)
    mask_indices = sorted(mask_indices)

    masked_token_labels = []
    for index in mask_indices:
        # 80% of the time, replace with [MASK]
        if random() < 0.8:
            masked_token = "[MASK]"
        else:
            # 10% of the time, keep original
            if random() < 0.5:
                masked_token = tokens[index]
            # 10% of the time, replace with random word
            else:
                masked_token = choice(vocab_list)
        masked_token_labels.append(tokens[index])
        # Once we've saved the true label for that token, we can overwrite it with the masked version
        tokens[index] = masked_token

    return tokens, mask_indices, masked_token_labels

def create_masked_norm_predictions(tokens, vocab_list, tokens_ent_mask, tokens_wp_to_ent, current_chunk_ent):
    tokens = copy.deepcopy(tokens)

    mask_indices = []
    is_ent_start = False
    start = -1
    for (i, token) in enumerate(tokens):
        if tokens_ent_mask[i] == 1 and token != "[CLS]" and token != "[SEP]":
            if is_ent_start == False:
                start = i
                is_ent_start = True
        else:
            if is_ent_start == True:
                mask_indices.append((start, i-1))
                start = -1
            is_ent_start = False

    mask_start_indices = []
    masked_token_labels = []
    for start, end in mask_indices:
        # 80% of the time, use the entity
        if random() < 0.8:
            pass
        else:
            # 15% of the time, mask the entity
            if random() < 0.75:
                j = start
                while j <= end:
                    tokens[j] = "[MASK]"
                    j += 1
            # 5% of the time, replace with other entity
            else:
                j = start
                while j <= end:
                    tokens[j] = choice(vocab_list)
                    j += 1

        mask_start_indices.append(start)
        masked_token_labels.append(current_chunk_ent[tokens_wp_to_ent[start]]['norm_id'])

    return tokens, mask_start_indices, masked_token_labels

def func1(sentence, tokenizer, current_chunk_ent):
    wp_to_orig_index = []
    orig_to_wp_index = []
    segment = []
    for token_idx, token in enumerate(sentence['text_token']):
        orig_to_wp_index.append(len(segment))
        word_pieces = tokenizer.tokenize(token[0])
        for word_piece in word_pieces:
            wp_to_orig_index.append(token_idx)
            segment.append(word_piece)

    segment_wp_to_ent = [-1] * len(segment)
    segment_ent_mask = [0] * len(segment)
    for idx, entity in enumerate(sentence['entity']):
        current_chunk_ent.append(entity)
        entity_tk_idx = entity['tk_start']
        while entity_tk_idx <= entity['tk_end']:
            for wp_to_orig_i, wp_to_orig_value in enumerate(wp_to_orig_index):
                if wp_to_orig_value == entity_tk_idx:
                    segment_ent_mask[wp_to_orig_i] = 1
                    segment_wp_to_ent[wp_to_orig_i] = len(current_chunk_ent)-1
            entity_tk_idx += 1


    return segment, segment_ent_mask, segment_wp_to_ent

def create_instances_from_document(
        doc_database, doc_idx, max_seq_length, short_seq_prob,
        masked_lm_prob, max_predictions_per_seq, vocab_list, tokenizer):
    """This code is mostly a duplicate of the equivalent function from Google BERT's repo.
    However, we make some changes and improvements. Sampling is improved and no longer requires a loop in this function.
    Also, documents are sampled proportionally to the number of sentences they contain, which means each sentence
    (rather than each document) has an equal chance of being sampled as a false example for the NextSentence task."""
    document = doc_database[doc_idx]
    # Account for [CLS], [SEP], [SEP]
    max_num_tokens = max_seq_length - 3

    # We *usually* want to fill up the entire sequence since we are padding
    # to `max_seq_length` anyways, so short sequences are generally wasted
    # computation. However, we *sometimes*
    # (i.e., short_seq_prob == 0.1 == 10% of the time) want to use shorter
    # sequences to minimize the mismatch between pre-training and fine-tuning.
    # The `target_seq_length` is just a rough target however, whereas
    # `max_seq_length` is a hard limit.
    target_seq_length = max_num_tokens
    if random() < short_seq_prob:
        target_seq_length = randint(2, max_num_tokens)

    # We DON'T just concatenate all of the tokens from a document into a long
    # sequence and choose an arbitrary split point because this would make the
    # next sentence prediction task too easy. Instead, we split the input into
    # segments "A" and "B" based on the actual "sentences" provided by the user
    # input.
    instances = []
    current_chunk = []
    current_length = 0

    current_chunk_ent = []
    current_chunk_ent_mask = []
    current_chunk_wp_to_ent = []

    i = 0
    while i < len(document):
        sentence = document[i]
        # segment = tokenizer.tokenize(sentence['text'])

        segment, segment_ent_mask, segment_wp_to_ent = func1(sentence, tokenizer, current_chunk_ent)

        current_chunk.append(segment)
        current_length += len(segment)
        current_chunk_ent_mask.append(segment_ent_mask)
        current_chunk_wp_to_ent.append(segment_wp_to_ent)
        if i == len(document) - 1 or current_length >= target_seq_length:
            if current_chunk:
                # `a_end` is how many segments from `current_chunk` go into the `A`
                # (first) sentence.
                a_end = 1
                if len(current_chunk) >= 2:
                    a_end = randint(1, len(current_chunk) - 1)

                tokens_a = []
                tokens_a_ent_mask = []
                tokens_a_wp_to_ent = []
                for j in range(a_end):
                    tokens_a.extend(current_chunk[j])
                    tokens_a_ent_mask.extend(current_chunk_ent_mask[j])
                    tokens_a_wp_to_ent.extend(current_chunk_wp_to_ent[j])


                tokens_b = []
                tokens_b_ent_mask = []
                tokens_b_wp_to_ent = []

                # Random next
                if len(current_chunk) == 1 or random() < 0.5:
                    is_random_next = True
                    target_b_length = target_seq_length - len(tokens_a)

                    # Sample a random document, with longer docs being sampled more frequently
                    random_document = doc_database.sample_doc(len(doc_database), current_idx=doc_idx, sentence_weighted=True)

                    random_start = randint(0, len(random_document) - 1)
                    for j in range(random_start, len(random_document)):

                        segment_tokens_b, segment_ent_mask_tokens_b, segment_wp_to_ent_tokens_b = func1(random_document[j], tokenizer, current_chunk_ent)

                        tokens_b.extend(segment_tokens_b)
                        tokens_b_ent_mask.extend(segment_ent_mask_tokens_b)
                        tokens_b_wp_to_ent.extend(segment_wp_to_ent_tokens_b)
                        if len(tokens_b) >= target_b_length:
                            break
                    # We didn't actually use these segments so we "put them back" so
                    # they don't go to waste.
                    num_unused_segments = len(current_chunk) - a_end
                    i -= num_unused_segments
                # Actual next
                else:
                    is_random_next = False
                    for j in range(a_end, len(current_chunk)):
                        tokens_b.extend(current_chunk[j])
                        tokens_b_ent_mask.extend((current_chunk_ent_mask[j]))
                        tokens_b_wp_to_ent.extend(current_chunk_wp_to_ent[j])
                truncate_seq_pair(tokens_a, tokens_b, max_num_tokens, tokens_a_ent_mask, tokens_b_ent_mask,
                                  tokens_a_wp_to_ent, tokens_b_wp_to_ent)

                # assert len(tokens_a) >= 1
                # assert len(tokens_b) >= 1
                # feili
                if len(tokens_a) < 1 or len(tokens_b) < 1:
                    if len(tokens_a) < 1:
                        logging.debug("create_instances_from_document: len(tokens_a) < 1: {}".format(tokens_a))
                    else:
                        logging.debug("create_instances_from_document: len(tokens_b) < 1: {}".format(tokens_b))
                    current_chunk = []
                    current_length = 0
                    current_chunk_ent = []
                    current_chunk_ent_mask = []
                    current_chunk_wp_to_ent = []
                    i += 1
                    continue

                tokens = ["[CLS]"] + tokens_a + ["[SEP]"] + tokens_b + ["[SEP]"]
                tokens_ent_mask = [1] + tokens_a_ent_mask + [1] + tokens_b_ent_mask + [1]
                tokens_wp_to_ent = [-1] + tokens_a_wp_to_ent + [-1] + tokens_b_wp_to_ent + [-1]
                # The segment IDs are 0 for the [CLS] token, the A tokens and the first [SEP]
                # They are 1 for the B tokens and the final [SEP]
                segment_ids = [0 for _ in range(len(tokens_a) + 2)] + [1 for _ in range(len(tokens_b) + 1)]

                # masked_tokens, masked_lm_positions, masked_lm_labels = create_masked_lm_predictions(
                #     tokens, masked_lm_prob, max_predictions_per_seq, vocab_list)

                try:
                    masked_tokens, masked_lm_positions, masked_lm_labels = create_masked_lm_and_ner_predictions(
                        tokens, masked_lm_prob, max_predictions_per_seq, vocab_list, tokens_ent_mask)
                except ValueError as err:
                    logging.info("ERROR: {}. TOKENS: {}".format(err, tokens))
                    current_chunk = []
                    current_length = 0
                    current_chunk_ent = []
                    current_chunk_ent_mask = []
                    current_chunk_wp_to_ent = []
                    i += 1
                    continue


                masked_tokens_ent, masked_start_ent, masked_norm_labels = create_masked_norm_predictions(tokens, vocab_list, tokens_ent_mask, tokens_wp_to_ent, current_chunk_ent)

                instance = {
                    "tokens": masked_tokens,
                    "segment_ids": segment_ids,
                    "is_random_next": is_random_next,
                    "masked_lm_positions": masked_lm_positions,
                    "masked_lm_labels": masked_lm_labels,
                    "tokens_ent": masked_tokens_ent,
                    "tokens_ent_mask": tokens_ent_mask,
                    "ent_start": masked_start_ent,
                    "norm_label": masked_norm_labels

                }
                instances.append(instance)

            current_chunk = []
            current_length = 0
            current_chunk_ent = []
            current_chunk_ent_mask = []
            current_chunk_wp_to_ent = []
        i += 1

    return instances

def prepare_instance(opt):

    tokenizer = BertTokenizer.from_pretrained(opt.bert_dir, do_lower_case=opt.do_lower_case)
    vocab_list = list(tokenizer.vocab.keys())
    with DocumentDatabase(reduce_memory=False) as docs:
        with open(opt.merged_file, 'r') as f:
            doc = []
            for line in tqdm(f, desc="Loading Dataset", unit=" lines"):
                line = line.strip()
                if line == "":
                    docs.add_document(doc)
                    doc = []
                else:
                    sentence = json.loads(line)
                    # tokens = tokenizer.tokenize(sentence['text'])
                    # doc.append(tokens)
                    doc.append(sentence)


        makedir_and_clear(opt.instance_dir)
        for epoch in trange(opt.iter, desc="Epoch"):
            epoch_filename = os.path.join(opt.instance_dir, f"epoch_{epoch}.json")
            num_instances = 0
            with open(epoch_filename, 'w') as epoch_file:
                for doc_idx in trange(len(docs), desc="Document"):
                    doc_instances = create_instances_from_document(
                        docs, doc_idx, max_seq_length=opt.max_seq_length, short_seq_prob=opt.short_seq_prob,
                        masked_lm_prob=opt.masked_lm_prob, max_predictions_per_seq=opt.max_predictions_per_seq,
                        vocab_list=vocab_list, tokenizer=tokenizer)
                    doc_instances = [json.dumps(instance) for instance in doc_instances]
                    for instance in doc_instances:
                        epoch_file.write(instance + '\n')
                        num_instances += 1
            metrics_file = os.path.join(opt.instance_dir, f"epoch_{epoch}_metrics.json")
            with open(metrics_file, 'w') as metrics_file:
                metrics = {
                    "num_training_examples": num_instances,
                    "max_seq_len": opt.max_seq_length
                }
                metrics_file.write(json.dumps(metrics))