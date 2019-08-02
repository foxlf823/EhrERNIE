import random
import logging
import numpy as np
import torch

from options import opt
import data
from my_utils import makedir_and_clear
import umls
from alphabet import Alphabet
from norm_utils import init_dict_alphabet, get_dict_index, get_dict_size, get_dict_name
from pytorch_pretrained_bert import BertTokenizer, BertAdam
tokenizer = BertTokenizer.from_pretrained(opt.bert_dir, do_lower_case=opt.do_lower_case)
from stopword import stop_word
from torch.utils.data import DataLoader, Dataset
import torch
import torch.nn as nn
from knowledge_bert.modeling import PreTrainedBertModel, BertModel
import time
import os
from data_structure import Entity

def normalize_word(word):
    new_word = ""
    for char in word:
        if char.isdigit():
            new_word += '0'
        else:
            new_word += char
    return new_word

def word_preprocess(word):
    if opt.norm_number_normalized:
        word = normalize_word(word)
    word = word.lower()
    return word

def generate_instances_ehr(entities, dict_alphabet, dictionary_reverse):
    Xs = []
    Ys = []

    for entity in entities:
        if len(entity.norm_ids) > 0:
            if entity.norm_ids[0] in dictionary_reverse:
                cui_list = dictionary_reverse[entity.norm_ids[0]]
                Y = get_dict_index(dict_alphabet, cui_list[0])  # use the first id to generate instance
                if Y >= 0 and Y < get_dict_size(dict_alphabet):
                    Ys.append(Y)
                else:
                    raise RuntimeError("entity {}, {}, cui not in dict_alphabet".format(entity.id, entity.name))
            else:
                logging.debug("entity {}, {}, can't map to umls, ignored".format(entity.id, entity.name))
                continue
        else:
            Ys.append(0)


        X = {}

        tokens = []
        for token in tokenizer.tokenize(entity.name):
            if token in stop_word:
                continue
            token = word_preprocess(token)
            tokens.append(token)

        tokens.insert(0, '[CLS]')
        tokens.append('[SEP]')
        word_ids = tokenizer.convert_tokens_to_ids(tokens)

        if len(word_ids) == 0:
            continue

        X['token'] = word_ids
        X['segment'] = [0]*len(word_ids)
        X['mask'] = [1] * len(word_ids)

        Xs.append(X)

    return Xs, Ys

class MyDataset(Dataset):

    def __init__(self, X, Y):
        self.X = X
        self.Y = Y

        assert len(self.X) == len(self.Y), 'X and Y have different lengths'

    def __len__(self):
        return len(self.Y)

    def __getitem__(self, idx):
        return (self.X[idx], self.Y[idx])

def pad_sequence(x, max_len):

    padded_x = np.zeros((len(x), max_len), dtype=np.int)
    for i, row in enumerate(x):
        padded_x[i][:len(row)] = row

    padded_x = torch.LongTensor(padded_x)

    return padded_x

import copy
def my_collate(batch):
    x, y = zip(*batch)

    tokens = [x_['token'] for x_ in x]
    lengths = [len(x_['token']) for x_ in x]
    max_len = max(lengths)
    mask = [x_['mask'] for x_ in x]
    sentences = [x_['segment'] for x_ in x]

    #since we didn't use context, tokens_ent and tokens are the same
    tokens_ent = copy.deepcopy(tokens)
    mask_ent = copy.deepcopy(mask)

    tokens = pad_sequence(tokens, max_len)
    mask = pad_sequence(mask, max_len)
    sentences = pad_sequence(sentences, max_len)

    tokens_ent = pad_sequence(tokens_ent, max_len)
    mask_ent = pad_sequence(mask_ent, max_len)

    y = torch.LongTensor(y).view(-1)

    lengths = torch.LongTensor(lengths)

    if opt.gpu >= 0 and torch.cuda.is_available():
        tokens = tokens.cuda(opt.gpu)
        mask = mask.cuda(opt.gpu)
        sentences = sentences.cuda(opt.gpu)
        y = y.cuda(opt.gpu)
        tokens_ent = tokens_ent.cuda(opt.gpu)
        mask_ent = mask_ent.cuda(opt.gpu)
    return tokens, mask, sentences, y, lengths, tokens_ent, mask_ent


class BertForSequenceClassification(PreTrainedBertModel):


    def __init__(self, config, target):
        super(BertForSequenceClassification, self).__init__(config)
        self.num_labels = get_dict_size(target)
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, self.num_labels)
        self.apply(self.init_bert_weights)
        self.dict_alphabet = target
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, input_ids, token_type_ids=None, attention_mask=None,
                input_ent = None, ent_mask = None):

        encoded_layers, pooled_output = self.bert(input_ids, token_type_ids, attention_mask, input_ent, ent_mask,
                                                   output_all_encoded_layers=False)
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        return encoded_layers, logits

    def loss(self, y_pred, y_gold):

        return self.criterion(y_pred, y_gold)

    def process_one_doc(self, doc, entities, dictionary, dictionary_reverse):

        Xs, Ys = generate_instances_ehr(entities, self.dict_alphabet, dictionary_reverse)

        data_loader = DataLoader(MyDataset(Xs, Ys), opt.batch_size, shuffle=False, collate_fn=my_collate)
        data_iter = iter(data_loader)
        num_iter = len(data_loader)

        entity_start = 0

        for i in range(num_iter):

            x, mask, sentences, _, _, tokens_ent, mask_ent = next(data_iter)

            _, y_pred = self.forward(x, sentences, mask, tokens_ent, mask_ent)

            values, indices = torch.max(y_pred, 1)

            actual_batch_size = x.size(0)

            for batch_idx in range(actual_batch_size):
                entity = entities[entity_start+batch_idx]
                norm_id = get_dict_name(self.dict_alphabet, indices[batch_idx].item())

                concept = dictionary[norm_id]
                entity.norm_ids.append(norm_id)
                entity.norm_names.append(concept.names)

            entity_start += actual_batch_size


def evaluate_for_ehr(gold_entities, pred_entities, dictionary):

    ct_norm_gold = len(gold_entities)
    ct_norm_predict = len(pred_entities)
    ct_norm_correct = 0



    for predict_entity in pred_entities:

        for gold_entity in gold_entities:

            if predict_entity.equals_span(gold_entity):

                if len(gold_entity.norm_ids) == 0:
                    # if gold_entity not annotated, we count it as TP
                    ct_norm_correct += 1
                else:

                    if len(predict_entity.norm_ids) != 0 and predict_entity.norm_ids[0] in dictionary:
                        concept = dictionary[predict_entity.norm_ids[0]]

                        if gold_entity.norm_ids[0] in concept.codes:
                            ct_norm_correct += 1


                break

    return ct_norm_gold, ct_norm_predict, ct_norm_correct

def evaluate(documents, dictionary, dictionary_reverse, model):
    model.eval()

    ct_predicted = 0
    ct_gold = 0
    ct_correct = 0

    for document in documents:

        # copy entities from gold entities
        pred_entities = []
        for gold in document.entities:
            pred = Entity()
            pred.id = gold.id
            pred.type = gold.type
            pred.spans = gold.spans
            pred.section = gold.section
            pred.name = gold.name
            pred_entities.append(pred)

        model.process_one_doc(document, pred_entities, dictionary, dictionary_reverse)

        p1, p2, p3 = evaluate_for_ehr(document.entities, pred_entities, dictionary)

        ct_gold += p1
        ct_predicted += p2
        ct_correct += p3


    if ct_gold == 0:
        precision = 0
        recall = 0
    else:
        precision = ct_correct * 1.0 / ct_predicted
        recall = ct_correct * 1.0 / ct_gold

    if precision+recall == 0:
        f_measure = 0
    else:
        f_measure = 2*precision*recall/(precision+recall)

    return precision, recall, f_measure

def train(train_data, dev_data, test_data, dictionary, dictionary_reverse, opt, fold_idx, isMeddra_dict):

    logging.info("build alphabet ...")

    dict_alphabet = Alphabet('dict')
    init_dict_alphabet(dict_alphabet, dictionary)
    dict_alphabet.close()

    train_X = []
    train_Y = []
    for doc in train_data:

        temp_X, temp_Y = generate_instances_ehr(doc.entities, dict_alphabet, dictionary_reverse)
        train_X.extend(temp_X)
        train_Y.extend(temp_Y)


    train_loader = DataLoader(MyDataset(train_X, train_Y), opt.batch_size, shuffle=True, collate_fn=my_collate)

    if opt.gpu >= 0 and torch.cuda.is_available():
        device = torch.device('cuda', opt.gpu)
    else:
        device = torch.device('cpu')

    model, _ = BertForSequenceClassification.from_pretrained(opt.bert_dir,
                                                          target=dict_alphabet)
    model.dict_alphabet = dict_alphabet
    model.to(device)

    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]

    optimizer = BertAdam(optimizer_grouped_parameters, lr=opt.lr)

    best_dev_f = -10
    best_dev_p = -10
    best_dev_r = -10

    bad_counter = 0

    logging.info("start training ...")

    for idx in range(opt.iter):
        epoch_start = time.time()

        model.train()

        train_iter = iter(train_loader)
        num_iter = len(train_loader)

        sum_loss = 0

        correct, total = 0, 0

        for i in range(num_iter):

            x, mask, sentences, y, _, tokens_ent, mask_ent = next(train_iter)

            _, y_pred = model.forward(x, sentences, mask, tokens_ent, mask_ent)

            l = model.loss(y_pred, y)

            sum_loss += l.item()

            l.backward()
            optimizer.step()
            model.zero_grad()

            total += y.size(0)
            _, pred = torch.max(y_pred, 1)
            correct += (pred == y).sum().item()

        epoch_finish = time.time()
        accuracy = 100.0 * correct / total
        logging.info("epoch: %s training finished. Time: %.2fs. loss: %.4f Accuracy %.2f" % (
        idx, epoch_finish - epoch_start, sum_loss / num_iter, accuracy))

        if opt.dev_file:
            p, r, f = evaluate(dev_data, dictionary, dictionary_reverse, model)
            logging.info("Dev: p: %.4f, r: %.4f, f: %.4f" % (p, r, f))
        else:
            f = best_dev_f

        if f > best_dev_f:
            logging.info("Exceed previous best f score on dev: %.4f" % (best_dev_f))

            torch.save(model, os.path.join(opt.save, "norm_neural.pkl"))

            best_dev_f = f
            best_dev_p = p
            best_dev_r = r

            bad_counter = 0
        else:
            bad_counter += 1

        if len(opt.dev_file) != 0 and bad_counter >= opt.patience:
            logging.info('Early Stop!')
            break

    logging.info("train finished")


    return best_dev_p, best_dev_r, best_dev_f

if __name__ == "__main__":


    logger = logging.getLogger()
    if opt.verbose:
        logger.setLevel(logging.DEBUG)
    else:
        logger.setLevel(logging.INFO)

    logging.info(opt)

    if opt.random_seed != 0:
        random.seed(opt.random_seed)
        np.random.seed(opt.random_seed)
        torch.manual_seed(opt.random_seed)
        torch.cuda.manual_seed_all(opt.random_seed)

    if opt.whattodo == 'train':

        config = data.read_config(opt.config)

        logging.info(config)

        makedir_and_clear(opt.save)

        logging.info("load data ...")
        train_data = data.loadData(opt.train_file, True, opt.types, opt.type_filter)
        dev_data = data.loadData(opt.dev_file, True, opt.types, opt.type_filter)
        if opt.test_file:
            test_data = data.loadData(opt.test_file, False, opt.types, opt.type_filter)
        else:
            test_data = None

        logging.info("load dict ...")
        UMLS_dict, UMLS_dict_reverse = umls.load_umls_MRCONSO(config['norm_dict'])
        logging.info("dict concept number {}".format(len(UMLS_dict)))

        train(train_data, dev_data, test_data, UMLS_dict, UMLS_dict_reverse, opt, None, False)

    else:
        raise RuntimeError("wrong whattodo {}".format(opt.whattodo))

    logging.info('done ........')

