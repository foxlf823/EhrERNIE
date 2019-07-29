
from my_utils import makedir_and_clear
import os
import re
import logging
import codecs
from data_structure import Document, Entity

def apply_metamap_to(input_dir, output_dir):
    makedir_and_clear(output_dir)

    for input_file_name in os.listdir(input_dir):
        input_file_path = os.path.join(input_dir, input_file_name)

        output_file_path = os.path.join(output_dir, input_file_name)
        os.system(
            '/Users/feili/tools/metamap/public_mm/bin/metamap -y -I -N --blanklines 0 -R SNOMEDCT_US,MDR -J acab,anab,comd,cgab,dsyn,emod,fndg,inpo,mobd,neop,patf,sosy {} {}'.format(
                input_file_path, output_file_path))

def load_metamap_result_from_file(file_path):
    re_brackets = re.compile(r'\[[0-9|/]+\]')
    document = Document()
    entities = []
    with codecs.open(file_path, 'r', 'UTF-8') as fp:
        for line in fp.readlines():
            fields = line.strip().split(u"|")

            if fields[1] != u'MMI':
                continue

            ID = fields[0] # Unique identifier used to identify text being processed. If no identifier is found in the text, 00000000 will be displayed
            MMI = fields[1] # Always MMI
            Score = fields[2] # MetaMap Indexing (MMI) score with a maximum score of 1000.00
            UMLS_Prefer_Name = fields[3] # The UMLS preferred name for the UMLS concept
            UMLS_ID = fields[4] # The CUI for the identified UMLS concept.
            Semantic_Type_List = fields[5] # Comma-separated list of Semantic Type abbreviations
            Trigger_Information = fields[6] # Comma separated sextuple showing what triggered MMI to identify this UMLS concept
            Location = fields[7] # Summarizes where UMLS concept was found. TI – Title, AB – Abstract, TX – Free Text, TI;AB – Title and Abstract
            Positional_Information = fields[8] # Semicolon-separated list of positional-information terns, showing StartPos, slash (/), and Length of each trigger identified in the Trigger Information field
            Treecode = fields[9] # Semicolon-separated list of any MeSH treecode


            triggers = Trigger_Information[1:-1].split(u",\"")
            spans = Positional_Information.split(u";")
            if len(triggers) != len(spans):
                raise RuntimeError("the number of triggers is not equal to that of spans: {} in {}".format(UMLS_ID, file_path[file_path.rfind('/')+1:]))

            for idx, span in enumerate(spans):
                bracket_spans = re_brackets.findall(span)
                if len(bracket_spans) == 0: # simple form
                    if span.find(u',') != -1:
                        logging.debug("ignore non-continuous form of Positional_Information: {} in {}".format(triggers[idx],
                                                                                                  file_path[
                                                                                                  file_path.rfind(
                                                                                                      '/') + 1:]))
                        continue


                    tmps = span.split(u"/")
                    entity = Entity()
                    entity.spans.append([int(tmps[0]), int(tmps[0]) + int(tmps[1])])
                    entity.norm_ids.append(str(UMLS_ID))
                    entity.norm_names.append(UMLS_Prefer_Name)
                    # "B cell lymphoma"-tx-5-"B cell lymphoma"-noun-0
                    tmps = triggers[idx].split(u"-")

                    if tmps[3].find('"') == -1:
                        logging.debug("ignore non-string entity: {} in {}".format(tmps[3], file_path[file_path.rfind('/') + 1:]))
                        continue


                    entity.name = tmps[3][1:-1] # remove ""

                    entities.append(entity)
                else:
                    for bracket_span in bracket_spans:
                        if bracket_span.find(u',') != -1:
                            logging.debug("ignore non-continuous form of Positional_Information: {} in {}".format(triggers[idx],
                                                                                                      file_path[
                                                                                                      file_path.rfind(
                                                                                                          '/') + 1:]))
                            continue

                        tmps = bracket_span[1:-1].split(u"/")
                        entity = Entity()
                        entity.spans.append([int(tmps[0]), int(tmps[0]) + int(tmps[1])])
                        entity.norm_ids.append(str(UMLS_ID))
                        entity.norm_names.append(UMLS_Prefer_Name)
                        # "B cell lymphoma"-tx-5-"B cell lymphoma"-noun-0
                        tmps = triggers[idx].split(u"-")

                        if tmps[3].find('"') == -1:
                            logging.debug("ignore non-string entity: {} in {}".format(tmps[3],
                                                                              file_path[
                                                                              file_path.rfind(
                                                                                  '/') + 1:]))
                            continue

                        entity.name = tmps[3][1:-1]

                        entities.append(entity)


    document.entities = entities
    return document

def find_entity_from_sentence(start, end, entities, sort):
    ret = []
    for entity in entities:
        if entity.spans[0][0] >= start and entity.spans[0][1] <= end:
            ret.append(entity)

    if sort:
        ret = sorted(ret, key=lambda x:x.spans[0][0])

    # transfer entity to dict
    dict_ret = []
    for entity in ret:
        dict_en = {'name':entity.name, 'norm_id':entity.norm_ids[0], 'norm_name':entity.norm_names[0],
                   'start':entity.spans[0][0]-start,'end':entity.spans[0][1]-start}
        dict_ret.append(dict_en)

    return dict_ret




from parseRecord import softLineBreak, sent_tokenizer, softLineBreak1, softLineBreak2
import nltk
import json
from my_utils import token_from_sent

def merge_text_and_metamap(text_dir, metamap_dir, out_file):

    out_fp = open(out_file, 'w')

    token_num = 0
    sentence_num = 0
    file_num = 0
    max_sentence_len = -1
    min_sentence_len = 9999

    for input_file_name in os.listdir(text_dir):
        with open(os.path.join(text_dir, input_file_name)) as text_fp:

            annotation = load_metamap_result_from_file(os.path.join(metamap_dir, input_file_name))


            whole_text = text_fp.read()
            paragraphs = softLineBreak2(whole_text)
            offset = 0
            for para in paragraphs:

                for st, en in sent_tokenizer.span_tokenize(para):
                    sent_text = para[st:en].replace('\n', ' ').replace("\t", " ")

                    sent_start = offset+st
                    sent_end = offset+en
                    entities = find_entity_from_sentence(sent_start, sent_end, annotation.entities, True)


                    tokens = token_from_sent(sent_text, 0)
                    t_num = len(tokens)

                    if t_num < 3 and len(entities) == 0:
                        continue

                    all_entity_find_tk = True
                    for entity in entities:
                        tk_start = -1
                        tk_end = -1
                        for token_idx, token in enumerate(tokens):
                            if token[1] == entity['start']:
                                tk_start = token_idx
                            if token[2] == entity['end']:
                                tk_end = token_idx
                            if tk_start != -1 and tk_end != -1:
                                break

                        if tk_start == -1 or tk_end == -1:
                            logging.debug('entity {}, tk_start {}, tk_end {}'.format(entity['name'], tk_start, tk_end))
                            all_entity_find_tk = False
                            break
                        else:
                            entity['tk_start'] = tk_start
                            entity['tk_end'] = tk_end

                    if all_entity_find_tk == False:
                        continue

                    sentence = {'text':sent_text,'entity':entities, 'text_token':tokens}
                    out_fp.write(json.dumps(sentence) + "\n")

                    if t_num < min_sentence_len:
                        min_sentence_len = t_num
                    if t_num > max_sentence_len:
                        max_sentence_len = t_num

                    token_num += t_num
                    sentence_num += 1

                offset += len(para)

        out_fp.write( "\n")
        file_num += 1


    print("token number: {}".format(token_num))
    print("sentence number: {}".format(sentence_num))
    print("max sentence length: {}".format(max_sentence_len))
    print("min sentence length: {}".format(min_sentence_len))
    print("file number: {}".format(file_num))

    out_fp.close()


