import os
import shutil
import re
import nltk

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

def text_tokenize(txt, sent_start):

    tokens=my_tokenize(txt)
    offset = 0
    for token in tokens:
        offset = txt.find(token, offset)
        yield token, offset+sent_start, offset+len(token)+sent_start
        offset += len(token)

def token_from_sent(txt, sent_start):
    return [token for token in text_tokenize(txt, sent_start)]


def makedir_and_clear(dir_path):
    if os.path.exists(dir_path):
        shutil.rmtree(dir_path)
        os.makedirs(dir_path)
    else:
        os.makedirs(dir_path)

def setList(listt, value):
    if (value not in listt) and (value != u""):
        listt.append(value)
    return listt