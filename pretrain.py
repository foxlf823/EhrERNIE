

import os
import json
from pathlib import Path
import torch
import logging
from my_utils import makedir_and_clear
from pytorch_pretrained_bert import BertTokenizer, BertAdam
from knowledge_bert.modeling import BertForPreTraining

from torch.utils.data import DataLoader, Dataset, RandomSampler
from torch.utils.data.distributed import DistributedSampler
from tempfile import TemporaryDirectory

from collections import namedtuple
import numpy as np
import umls
from alphabet import Alphabet
from norm_utils import init_dict_alphabet, get_dict_index, get_dict_size
import time

InputFeatures = namedtuple("InputFeatures", "input_ids input_mask segment_ids lm_label_ids is_next input_ids_ent input_mask_ent norm_label_ids")

def convert_example_to_features(example, tokenizer, max_seq_length, dict_alphabet):
    tokens = example["tokens"]
    segment_ids = example["segment_ids"]
    is_random_next = example["is_random_next"]
    masked_lm_positions = example["masked_lm_positions"]
    masked_lm_labels = example["masked_lm_labels"]

    tokens_ent = example['tokens_ent']
    tokens_ent_mask = example ['tokens_ent_mask']
    ent_start = example['ent_start']
    norm_label = example['norm_label']

    assert len(tokens) == len(segment_ids) <= max_seq_length  # The preprocessed data should be already truncated
    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    masked_label_ids = tokenizer.convert_tokens_to_ids(masked_lm_labels)

    input_array = np.zeros(max_seq_length, dtype=np.int)
    input_array[:len(input_ids)] = input_ids

    mask_array = np.zeros(max_seq_length, dtype=np.bool)
    mask_array[:len(input_ids)] = 1

    segment_array = np.zeros(max_seq_length, dtype=np.bool)
    segment_array[:len(segment_ids)] = segment_ids

    lm_label_array = np.full(max_seq_length, dtype=np.int, fill_value=-1)
    lm_label_array[masked_lm_positions] = masked_label_ids

    input_ids_ent = tokenizer.convert_tokens_to_ids(tokens_ent)
    norm_label_ids = [get_dict_index(dict_alphabet, l) for l in norm_label]

    input_array_ent = np.zeros(max_seq_length, dtype=np.int)
    input_array_ent[:len(input_ids_ent)] = input_ids_ent

    mask_array_ent = np.zeros(max_seq_length, dtype=np.bool)
    mask_array_ent[:len(tokens_ent_mask)] = tokens_ent_mask

    norm_label_array = np.full(max_seq_length, dtype=np.int, fill_value=-1)
    norm_label_array[ent_start] = norm_label_ids

    features = InputFeatures(input_ids=input_array,
                             input_mask=mask_array,
                             segment_ids=segment_array,
                             lm_label_ids=lm_label_array,
                             is_next=is_random_next,
                             input_ids_ent = input_array_ent,
                             input_mask_ent = mask_array_ent,
                             norm_label_ids = norm_label_array)
    return features


class PregeneratedDataset(Dataset):
    def __init__(self, training_path, epoch, tokenizer, num_data_epochs, dict_alphabet):
        self.vocab = tokenizer.vocab
        self.tokenizer = tokenizer
        self.epoch = epoch
        self.data_epoch = epoch % num_data_epochs
        data_file = training_path / f"epoch_{self.data_epoch}.json"
        metrics_file = training_path / f"epoch_{self.data_epoch}_metrics.json"
        assert data_file.is_file() and metrics_file.is_file()
        metrics = json.loads(metrics_file.read_text())
        num_samples = metrics['num_training_examples']
        seq_len = metrics['max_seq_len']
        self.temp_dir = None
        self.working_dir = None
        self.dict_alphabet = dict_alphabet

        input_ids = np.zeros(shape=(num_samples, seq_len), dtype=np.int32)
        input_masks = np.zeros(shape=(num_samples, seq_len), dtype=np.bool)
        segment_ids = np.zeros(shape=(num_samples, seq_len), dtype=np.bool)
        lm_label_ids = np.full(shape=(num_samples, seq_len), dtype=np.int32, fill_value=-1)
        is_nexts = np.zeros(shape=(num_samples,), dtype=np.bool)
        input_ids_ent = np.zeros(shape=(num_samples, seq_len), dtype=np.int32)
        input_mask_ent = np.zeros(shape=(num_samples, seq_len), dtype=np.bool)
        norm_label_ids = np.full(shape=(num_samples, seq_len), dtype=np.int32, fill_value=-1)

        logging.info(f"Loading training examples for epoch {epoch}")
        with data_file.open() as f:
            # for i, line in enumerate(tqdm(f, total=num_samples, desc="Training examples")):
            # feili
            for i, line in enumerate(f):
                line = line.strip()
                example = json.loads(line)
                features = convert_example_to_features(example, tokenizer, seq_len, self.dict_alphabet)
                input_ids[i] = features.input_ids
                segment_ids[i] = features.segment_ids
                input_masks[i] = features.input_mask
                lm_label_ids[i] = features.lm_label_ids
                is_nexts[i] = features.is_next
                input_ids_ent[i] = features.input_ids_ent
                input_mask_ent[i] = features.input_mask_ent
                norm_label_ids[i] = features.norm_label_ids
        assert i == num_samples - 1  # Assert that the sample count metric was true
        logging.info("Loading complete!")
        self.num_samples = num_samples
        self.seq_len = seq_len
        self.input_ids = input_ids
        self.input_masks = input_masks
        self.segment_ids = segment_ids
        self.lm_label_ids = lm_label_ids
        self.is_nexts = is_nexts
        self.input_ids_ent = input_ids_ent
        self.input_mask_ent = input_mask_ent
        self.norm_label_ids = norm_label_ids

    def __len__(self):
        return self.num_samples

    def __getitem__(self, item):
        return (torch.tensor(self.input_ids[item].astype(np.int64)),
                torch.tensor(self.input_masks[item].astype(np.int64)),
                torch.tensor(self.segment_ids[item].astype(np.int64)),
                torch.tensor(self.lm_label_ids[item].astype(np.int64)),
                torch.tensor(self.is_nexts[item].astype(np.int64)),
                torch.tensor(self.input_ids_ent[item].astype(np.int64)),
                torch.tensor(self.input_mask_ent[item].astype(np.int64)),
                torch.tensor(self.norm_label_ids[item].astype(np.int64))
                )


def pretrain(opt):

    samples_per_epoch = []
    pregenerated_data = Path(opt.instance_dir)
    for i in range(opt.iter):

        epoch_file = pregenerated_data / f"epoch_{i}.json"
        metrics_file = pregenerated_data / f"epoch_{i}_metrics.json"
        if epoch_file.is_file() and metrics_file.is_file():
            metrics = json.loads(metrics_file.read_text())
            samples_per_epoch.append(metrics['num_training_examples'])
        else:
            if i == 0:
                exit("No training data was found!")
            print(f"Warning! There are fewer epochs of pregenerated data ({i}) than training epochs ({opt.iter}).")
            print("This script will loop over the available data, but training diversity may be negatively impacted.")
            num_data_epochs = i
            break
    else:
        num_data_epochs = opt.iter

    if opt.gpu >= 0 and torch.cuda.is_available():
        if opt.multi_gpu:
            device = torch.device("cuda")
            n_gpu = torch.cuda.device_count()
        else:
            device = torch.device('cuda', opt.gpu)
            n_gpu = 1
    else:
        device = torch.device("cpu")
        n_gpu = 0

    logging.info("device: {} n_gpu: {}".format(device, n_gpu))

    if opt.gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
                            opt.gradient_accumulation_steps))

    opt.batch_size = opt.batch_size // opt.gradient_accumulation_steps

    makedir_and_clear(opt.save)

    tokenizer = BertTokenizer.from_pretrained(opt.bert_dir, do_lower_case=opt.do_lower_case)

    total_train_examples = 0
    for i in range(opt.iter):
        # The modulo takes into account the fact that we may loop over limited epochs of data
        total_train_examples += samples_per_epoch[i % len(samples_per_epoch)]

    num_train_optimization_steps = int(
        total_train_examples / opt.batch_size / opt.gradient_accumulation_steps)

    logging.info("load dict ...")
    UMLS_dict, UMLS_dict_reverse = umls.load_umls_MRCONSO(opt.norm_dict)
    logging.info("dict concept number {}".format(len(UMLS_dict)))
    dict_alphabet = Alphabet('dict')
    init_dict_alphabet(dict_alphabet, UMLS_dict)
    dict_alphabet.close()

    # Prepare model
    model, _ = BertForPreTraining.from_pretrained(opt.bert_dir, num_norm_labels=get_dict_size(dict_alphabet))
    model.to(device)
    if opt.multi_gpu > 1:
        model = torch.nn.DataParallel(model)


    # Prepare optimizer
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
         'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]

    optimizer = BertAdam(optimizer_grouped_parameters,
                         lr=opt.lr,
                         warmup=opt.warmup_proportion,
                         t_total=num_train_optimization_steps)



    global_step = 0
    logging.info("***** Running training *****")
    logging.info(f"  Num examples = {total_train_examples}")
    logging.info("  Batch size = %d", opt.batch_size)
    logging.info("  Num steps = %d", num_train_optimization_steps)
    model.train()
    for epoch in range(opt.iter):
        epoch_dataset = PregeneratedDataset(epoch=epoch, training_path=pregenerated_data, tokenizer=tokenizer,
                                            num_data_epochs=num_data_epochs, dict_alphabet=dict_alphabet)
        train_sampler = RandomSampler(epoch_dataset)

        train_dataloader = DataLoader(epoch_dataset, sampler=train_sampler, batch_size=opt.batch_size)
        tr_loss = 0
        nb_tr_examples, nb_tr_steps = 0, 0

        epoch_start = time.time()
        sum_loss = 0
        sum_orginal_loss = 0
        num_iter = len(train_dataloader)

        for step, batch in enumerate(train_dataloader):
            batch = tuple(t.to(device) for t in batch)
            input_ids, input_mask, segment_ids, lm_label_ids, is_next, input_ids_ent, input_mask_ent, norm_label_ids = batch
            loss, original_loss = model(input_ids, segment_ids, input_mask, lm_label_ids, input_ids_ent, input_mask_ent, is_next, norm_label_ids)
            if n_gpu > 1:
                loss = loss.mean() # mean() to average on multi-gpu.
                original_loss = original_loss.mean()
            if opt.gradient_accumulation_steps > 1:
                loss = loss / opt.gradient_accumulation_steps
                original_loss = original_loss/ opt.gradient_accumulation_steps

            loss.backward()

            if (step + 1) % opt.gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
                global_step += 1

            sum_loss += loss.item()
            sum_orginal_loss += original_loss.item()

        epoch_finish = time.time()
        logging.info("epoch: %s training finished. Time: %.2fs. loss: %.4f, original_loss %.4f" % (
        epoch, epoch_finish - epoch_start, sum_loss / num_iter, sum_orginal_loss / num_iter))

    # Save a trained model
    logging.info("** ** * Saving fine-tuned model ** ** * ")
    model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
    output_model_file = os.path.join(opt.save, "pytorch_model.bin")
    torch.save(model_to_save.state_dict(), str(output_model_file))