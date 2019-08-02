import codecs
from alphabet import Alphabet
import numpy as np
import pickle as pk
from os import listdir
from os.path import isfile, join
# import spacy
from data_structure import Entity, Document
from options import opt
import logging
import re
import nltk
import bioc

import json
import xml.sax


def getLabel(start, end, entities):
    match = ""
    for entity in entities:
        if start == entity.spans[0][0] and end == entity.spans[0][1] : # S
            match = "S"
            break
        elif start == entity.spans[0][0] and end != entity.spans[0][1] : # B
            match = "B"
            break
        elif start != entity.spans[0][0] and end == entity.spans[0][1] : # E
            match = "E"
            break
        elif start > entity.spans[0][0] and end < entity.spans[0][1]:  # M
            match = "M"
            break

    if match != "":
        if opt.no_type:
            return match + "-" +"X"
        else:
            return match+"-"+entity.type
    else:
        return "O"




def get_start_and_end_offset_of_token_from_spacy(token):
    start = token.idx
    end = start + len(token)
    return start, end

def get_sentences_and_tokens_from_spacy(text, spacy_nlp, entities):
    document = spacy_nlp(text)
    # sentences
    sentences = []
    for span in document.sents:
        sentence = [document[i] for i in range(span.start, span.end)]
        sentence_tokens = []
        for token in sentence:
            token_dict = {}
            token_dict['start'], token_dict['end'] = get_start_and_end_offset_of_token_from_spacy(token)
            token_dict['text'] = text[token_dict['start']:token_dict['end']]
            if token_dict['text'].strip() in ['\n', '\t', ' ', '']:
                continue
            # Make sure that the token text does not contain any space
            if len(token_dict['text'].split(' ')) != 1:
                logging.warning("the text of the token contains space character, replaced with hyphen\n\t{0}\n\t{1}".format(token_dict['text'],
                                                                                                                           token_dict['text'].replace(' ', '-')))
                token_dict['text'] = token_dict['text'].replace(' ', '-')

            # get label
            if entities is not None:
            # if entities:
                token_dict['label'] = getLabel(token_dict['start'], token_dict['end'], entities)

            sentence_tokens.append(token_dict)
        sentences.append(sentence_tokens)
    return sentences

pattern = re.compile(r'[-_/]+')

def my_split(s):
    text = []
    iter = re.finditer(pattern, s)
    start = 0
    for i in iter:
        if start != i.start():
            text.append(s[start: i.start()])
        text.append(s[i.start(): i.end()])
        start = i.end()
    if start != len(s):
        text.append(s[start: ])
    return text

def my_tokenize(txt):
    tokens1 = nltk.word_tokenize(txt.replace('"', " "))  # replace due to nltk transfer " to other character, see https://github.com/nltk/nltk/issues/1630
    tokens2 = []
    for token1 in tokens1:
        token2 = my_split(token1)
        tokens2.extend(token2)
    return tokens2

# if add pos, add to the end, so external functions don't need to be modified too much
# def text_tokenize_and_postagging(txt, sent_start):
#     tokens= my_tokenize(txt)
#     pos_tags = nltk.pos_tag(tokens)
#
#     offset = 0
#     for token, pos_tag in pos_tags:
#         offset = txt.find(token, offset)
#         yield token, pos_tag, offset+sent_start, offset+len(token)+sent_start
#         offset += len(token)

def text_tokenize_and_postagging(txt, sent_start):
    tokens= my_tokenize(txt)
    pos_tags = nltk.pos_tag(tokens)

    offset = 0
    for token, pos_tag in pos_tags:
        offset = txt.find(token, offset)
        yield token, offset+sent_start, offset+len(token)+sent_start, pos_tag
        offset += len(token)

def token_from_sent(txt, sent_start):
    return [token for token in text_tokenize_and_postagging(txt, sent_start)]

def get_sentences_and_tokens_from_nltk(text, nlp_tool, entities, ignore_regions, section_id):
    all_sents_inds = []
    generator = nlp_tool.span_tokenize(text)
    for t in generator:
        all_sents_inds.append(t)

    sentences = []
    for ind in range(len(all_sents_inds)):
        t_start = all_sents_inds[ind][0]
        t_end = all_sents_inds[ind][1]

        tmp_tokens = token_from_sent(text[t_start:t_end], t_start)
        sentence_tokens = []
        for token_idx, token in enumerate(tmp_tokens):
            token_dict = {}
            token_dict['start'], token_dict['end'] = token[1], token[2]
            token_dict['text'] = token[0]
            token_dict['pos'] = token[3]
            token_dict['cap'] = featureCapital(token[0])
            if token_dict['text'].strip() in ['\n', '\t', ' ', '']:
                continue
            # Make sure that the token text does not contain any space
            if len(token_dict['text'].split(' ')) != 1:
                logging.warning("the text of the token contains space character, replaced with hyphen\n\t{0}\n\t{1}".format(token_dict['text'],
                                                                                                                           token_dict['text'].replace(' ', '-')))
                token_dict['text'] = token_dict['text'].replace(' ', '-')

            # get label
            if entities is not None:
                token_dict['label'] = getLabel(token_dict['start'], token_dict['end'], entities)


            sentence_tokens.append(token_dict)

        sentences.append(sentence_tokens)
    return sentences



def get_text_file(filename):
    file = codecs.open(filename, 'r', 'UTF-8')
    data = file.read()
    file.close()
    return data

# bioc will try to use str even if you feed it with utf-8.
# if bioc can't use str to denote something, it will use unicode
def get_bioc_file(filename):
    with codecs.open(filename, 'r', 'UTF-8') as fp:
        data = fp.read()
        collection = bioc.loads(data)
        return collection.documents

def processOneFile(fileName, annotation_dir, corpus_dir, nlp_tool, isTraining, types, type_filter):
    document = Document()
    document.name = fileName[:fileName.find('.')]


    ct_snomed = 0
    ct_meddra = 0
    ct_unnormed = 0

    if annotation_dir:
        annotation_file = get_bioc_file(join(annotation_dir, fileName))
        bioc_passage = annotation_file[0].passages[0]
        entities = []

        for entity in bioc_passage.annotations:
            if types and (entity.infons['type'] not in type_filter):
                continue
            entity_ = Entity()
            entity_.id = entity.id
            processed_name = entity.text.replace('\\n', ' ')
            if len(processed_name) == 0:
                logging.debug("{}: entity {} name is empty".format(fileName, entity.id))
                continue
            entity_.name = processed_name

            entity_.type = entity.infons['type']
            entity_.spans.append([entity.locations[0].offset,entity.locations[0].end])
            if ('SNOMED code' in entity.infons and entity.infons['SNOMED code'] != 'N/A')\
                    and ('SNOMED term' in entity.infons and entity.infons['SNOMED term'] != 'N/A'):
                entity_.norm_ids.append(entity.infons['SNOMED code'])
                entity_.norm_names.append(entity.infons['SNOMED term'])
                ct_snomed += 1
            elif ('MedDRA code' in entity.infons and entity.infons['MedDRA code'] != 'N/A')\
                    and ('MedDRA term' in entity.infons and entity.infons['MedDRA term'] != 'N/A'):
                entity_.norm_ids.append(entity.infons['MedDRA code'])
                entity_.norm_names.append(entity.infons['MedDRA term'])
                ct_meddra += 1
            else:
                logging.debug("{}: no norm id in entity {}".format(fileName, entity.id))
                ct_unnormed += 1
                continue

            entities.append(entity_)

        document.entities = entities

    corpus_file = get_text_file(join(corpus_dir, fileName.split('.bioc')[0]))
    document.text = corpus_file

    if isTraining:
        sentences = get_sentences_and_tokens_from_nltk(corpus_file, nlp_tool, document.entities, None, None)
    else:
        sentences = get_sentences_and_tokens_from_nltk(corpus_file, nlp_tool, None, None, None)

    document.sentences = sentences

    return document, ct_snomed, ct_meddra, ct_unnormed




def loadData(basedir, isTraining, types, type_filter):

    logging.info("loadData: {}".format(basedir))

    list_dir = listdir(basedir)
    if 'bioc' in list_dir:
        annotation_dir = join(basedir, 'bioc')
    elif 'annotations' in list_dir:
        annotation_dir = join(basedir, 'annotations')
    else:
        raise RuntimeError("no bioc or annotations in {}".format(basedir))

    if 'txt' in list_dir:
        corpus_dir = join(basedir, 'txt')
    elif 'corpus' in list_dir:
        corpus_dir = join(basedir, 'corpus')
    else:
        raise RuntimeError("no txt or corpus in {}".format(basedir))

    nlp_tool = nltk.data.load('tokenizers/punkt/english.pickle')

    documents = []

    count_document = 0
    count_sentence = 0
    count_entity = 0
    count_entity_snomed = 0
    count_entity_meddra = 0
    count_entity_without_normed = 0

    annotation_files = [f for f in listdir(annotation_dir) if isfile(join(annotation_dir, f)) and f.find(".xml")!=-1]
    for fileName in annotation_files:
        try:
            document, p1, p2, p3 = processOneFile(fileName, annotation_dir, corpus_dir, nlp_tool, isTraining, types, type_filter)
        except Exception as e:
            logging.error("process file {} error: {}".format(fileName, e))
            continue

        documents.append(document)

        # statistics
        count_document += 1
        count_sentence += len(document.sentences)
        count_entity += len(document.entities)
        count_entity_snomed += p1
        count_entity_meddra += p2
        count_entity_without_normed += p3

    logging.info("document number: {}".format(count_document))
    logging.info("sentence number: {}".format(count_sentence))
    logging.info("entity number {}, snomed {}, meddra {}, unnormed {}".format(count_entity, count_entity_snomed,
                                                                              count_entity_meddra, count_entity_without_normed))

    return documents




# def _readString(f):
#     s = str()
#     c = f.read(1).decode('iso-8859-1')
#     while c != '\n' and c != ' ':
#         s = s + c
#         c = f.read(1).decode('iso-8859-1')
#
#     return s

def _readString(f, code):
    # s = unicode()
    s = str()
    c = f.read(1)
    value = ord(c)

    while value != 10 and value != 32:
        if 0x00 < value < 0xbf:
            continue_to_read = 0
        elif 0xC0 < value < 0xDF:
            continue_to_read = 1
        elif 0xE0 < value < 0xEF:
            continue_to_read = 2
        elif 0xF0 < value < 0xF4:
            continue_to_read = 3
        else:
            raise RuntimeError("not valid utf-8 code")

        i = 0
        # temp = str()
        # temp = temp + c

        temp = bytes()
        temp = temp + c

        while i<continue_to_read:
            temp = temp + f.read(1)
            i += 1

        temp = temp.decode(code)
        s = s + temp

        c = f.read(1)
        value = ord(c)

    return s

import struct
def _readFloat(f):
    bytes4 = f.read(4)
    f_num = struct.unpack('f', bytes4)[0]
    return f_num

def load_pretrain_emb(embedding_path):
    embedd_dim = -1
    embedd_dict = dict()
    # emb_debug = []
    if embedding_path.find('.bin') != -1:
        with open(embedding_path, 'rb') as f:
            wordTotal = int(_readString(f, 'utf-8'))
            embedd_dim = int(_readString(f, 'utf-8'))

            for i in range(wordTotal):
                word = _readString(f, 'utf-8')
                # emb_debug.append(word)

                word_vector = []
                for j in range(embedd_dim):
                    word_vector.append(_readFloat(f))
                word_vector = np.array(word_vector, np.float)

                f.read(1)  # a line break
                # try:
                #     embedd_dict[word.decode('utf-8')] = word_vector
                # except Exception , e:
                #     pass
                embedd_dict[word] = word_vector
    else:
        with codecs.open(embedding_path, 'r', 'UTF-8') as file:
        # with open(embedding_path, 'r') as file:
            for line in file:
                line = line.strip()
                if len(line) == 0:
                    continue
                tokens = line.split()
                # feili
                if len(tokens) == 2:
                    continue # it's a head
                if embedd_dim < 0:
                    embedd_dim = len(tokens) - 1
                else:
                    assert (embedd_dim + 1 == len(tokens))
                embedd = np.zeros([1, embedd_dim])
                embedd[:] = tokens[1:]
                embedd_dict[tokens[0]] = embedd
                # embedd_dict[tokens[0].decode('utf-8')] = embedd

    return embedd_dict, embedd_dim

def norm2one(vec):
    root_sum_square = np.sqrt(np.sum(np.square(vec)))
    return vec/root_sum_square

def build_pretrain_embedding(embedding_path, word_alphabet, embedd_dim, norm):
    embedd_dict = dict()
    if embedding_path != None:
        embedd_dict, embedd_dim = load_pretrain_emb(embedding_path)
    alphabet_size = word_alphabet.size()
    scale = np.sqrt(3.0 / embedd_dim)
    pretrain_emb = np.zeros([word_alphabet.size(), embedd_dim])
    perfect_match = 0
    case_match = 0
    digits_replaced_with_zeros_found = 0
    lowercase_and_digits_replaced_with_zeros_found = 0
    not_match = 0
    for word, index in word_alphabet.iteritems():
        if word in embedd_dict:
            if norm:
                pretrain_emb[index,:] = norm2one(embedd_dict[word])
            else:
                pretrain_emb[index,:] = embedd_dict[word]
            perfect_match += 1
        elif word.lower() in embedd_dict:
            if norm:
                pretrain_emb[index,:] = norm2one(embedd_dict[word.lower()])
            else:
                pretrain_emb[index,:] = embedd_dict[word.lower()]
            case_match += 1
        elif re.sub('\d', '0', word) in embedd_dict:
            if norm:
                pretrain_emb[index,:] = norm2one(embedd_dict[re.sub('\d', '0', word)])
            else:
                pretrain_emb[index,:] = embedd_dict[re.sub('\d', '0', word)]
            digits_replaced_with_zeros_found += 1
        elif re.sub('\d', '0', word.lower()) in embedd_dict:
            if norm:
                pretrain_emb[index,:] = norm2one(embedd_dict[re.sub('\d', '0', word.lower())])
            else:
                pretrain_emb[index,:] = embedd_dict[re.sub('\d', '0', word.lower())]
            lowercase_and_digits_replaced_with_zeros_found += 1
        else:
            pretrain_emb[index,:] = np.random.uniform(-scale, scale, [1, embedd_dim])
            not_match += 1
    pretrained_size = len(embedd_dict)
    logging.info("Embedding:\n     pretrain word:%s, prefect match:%s, case_match:%s, dig_zero_match:%s, "
                 "case_dig_zero_match:%s, oov:%s, oov%%:%s"
                 %(pretrained_size, perfect_match, case_match, digits_replaced_with_zeros_found,
                   lowercase_and_digits_replaced_with_zeros_found, not_match, (not_match+0.)/alphabet_size))
    return pretrain_emb, embedd_dim




def read_config(config_file):

    config = config_file_to_dict(config_file)
    return config

def config_file_to_dict(input_file):
    config = {}
    fins = open(input_file, 'r').readlines()
    for line in fins:
        line = line.strip()
        if line == '':
            continue
        if len(line) > 0 and line[0] == "#":
            continue

        pairs = line.split()
        if len(pairs) > 1:
            for idx, pair in enumerate(pairs):
                if idx == 0:
                    items = pair.split('=')
                    if items[0] not in config:
                        feat_dict = {}
                        config[items[0]] = feat_dict
                    feat_dict = config[items[0]]
                    feat_name = items[1]
                    one_dict = {}
                    feat_dict[feat_name] = one_dict
                else:
                    items = pair.split('=')
                    one_dict[items[0]] = items[1]
        else:
            items = pairs[0].split('=')
            if items[0] in config:
                print("Warning: duplicated config item found: %s, updated." % (items[0]))
            config[items[0]] = items[-1]

    return config

# def config_file_to_dict(input_file):
#     config = {}
#     fins = open(input_file, 'r').readlines()
#     for line in fins:
#         if len(line) > 0 and line[0] == "#":
#             continue
#         if "=" in line:
#             pair = line.strip().split('#', 1)[0].split('=', 1)
#             item = pair[0]
#             if item == "ner_feature":
#                 if item not in config:
#                     feat_dict = {}
#                     config[item] = feat_dict
#                 feat_dict = config[item]
#                 new_pair = pair[-1].split()
#                 feat_name = new_pair[0]
#                 one_dict = {}
#                 one_dict["emb_dir"] = None
#                 one_dict["emb_size"] = 10
#                 one_dict["emb_norm"] = False
#                 if len(new_pair) > 1:
#                     for idx in range(1, len(new_pair)):
#                         conf_pair = new_pair[idx].split('=')
#                         if conf_pair[0] == "emb_dir":
#                             one_dict["emb_dir"] = conf_pair[-1]
#                         elif conf_pair[0] == "emb_size":
#                             one_dict["emb_size"] = int(conf_pair[-1])
#                         elif conf_pair[0] == "emb_norm":
#                             one_dict["emb_norm"] = str2bool(conf_pair[-1])
#                 feat_dict[feat_name] = one_dict
#                 # print "feat",feat_dict
#             elif item == "ext_corpus":
#                 if item not in config:
#                     feat_dict = {}
#                     config[item] = feat_dict
#                 feat_dict = config[item]
#                 new_pair = pair[-1].split()
#                 feat_name = new_pair[0]
#                 one_dict = {}
#                 if len(new_pair) > 1:
#                     for idx in range(1, len(new_pair)):
#                         conf_pair = new_pair[idx].split('=')
#                         if conf_pair[0] == 'types':
#                             one_dict[conf_pair[0]] = set(conf_pair[1].split(','))
#                         else:
#                             one_dict[conf_pair[0]] = conf_pair[1]
#                 feat_dict[feat_name] = one_dict
#             else:
#                 if item in config:
#                     print("Warning: duplicated config item found: %s, updated." % (pair[0]))
#                 config[item] = pair[-1]
#     return config

def str2bool(string):
    if string == "True" or string == "true" or string == "TRUE":
        return True
    else:
        return False


def featureCapital(word):
    if word[0].isalpha() and word[0].isupper():
        return "1"
    else:
        return "0"




