import sys
import re
import os
import nltk

sent_tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
# python parseRecord.py input output



reservedPats = list(map(lambda pat: re.compile(pat), [
        r'.*PK\/PK$'
        , r'TC1'
        , r'_+$'
        , r'-+$'
        , r'\*.+\*'
        , r'\*\*DATE'
        , r'\*\*INITIALS'
        , r'.*\/\w{2,4}$'        
]))

def isReserved(line):
    lines = line.strip()
    return any([pat.match(lines) != None for pat in reservedPats])

def isSection(line):
    striped = line.strip()
    return striped != '' and (striped[-1] == ':' or re.match(r'^[A-Z\s]+$', striped) is not None)

tableRow = [ '    ', ': ', ' - ']
tablePatMulti = list(map(lambda pat: re.compile(r'(?P<first>[A-Z0-9].*)' + pat + r'(?P<rest>.+)'), [
        #  r': '
         r' - '
        , r'    '
        # , r'[A-Za-z]+:'
]))
tablePatSolo = list(map(lambda pat: re.compile(r'(?P<first>[A-Z0-9][^:]*)' + pat + r'(?P<rest>.+)'), [
         r': '
]))
tablePat = tablePatMulti + tablePatSolo
def isTableRow(line):
    striped = line.strip()
    return any([s.match(striped) != None for s in tablePat])

def isSpecialSent(line):
    return isReserved(line) or isSection(line)

def isAppendable(line):
    return isTableRow(line) or isSpecialSent(line)

def checkAny(preds):
    def check(line):
        return any([p(line) for p in preds])
    return check

listPats = list(map(lambda pat: re.compile(r'\s*'+pat), [
        r'[A-Z]\.\s*[A-Z]'
        , r'[0-9]+\.\s*[A-Z]'
]))
def isList(line):
    return any([pat.match(line) != None for pat in listPats])

def isCompleteTable(line):
    return isTableRow(line) and (len(line) < 40)

def parseTableRow(line):
    for s in tablePat:
        if s in tablePatSolo:
            fun = lambda x: [x]
        else:
            fun = parseTableRow
        res = s.search(line)
        if res != None:
            k = fun(res.group('first').strip())
            v = fun(res.group('rest').strip())
            return k + v
    return [line]


def couldJoin(pre, nex):
    if isCompleteTable(pre):
        return False
    if nex == '' or pre == '':
        return False
    joined = sent_tokenizer.sentences_from_text(pre + '\n' + nex)
    sents = sent_tokenizer.sentences_from_text(pre)
    return (sents != []) and (sents[-1] != joined[len(sents) - 1])

def appendable(pre, nex):
    return (not isSpecialSent(pre)) and (not checkAny([isTableRow, isReserved, isList])(nex)) and couldJoin(pre, nex)


# def addTag(line):
#     def getTag(line):
#         if isList(line):
#             return 'list'
#         if isTableRow(line):
#             return 'table'
#         if isReserved(line):
#             return 'reserved'
#         if isSection(line):
#             return 'section'
#         return 'text'

#     return (getTag(line), line)


def softLineBreak(text):
    lines = text.split('\n')
    paras = [lines[0]]
    for i in range(1, len(lines)):
        previous = paras[-1]
        if appendable(previous, lines[i]):
                previous += ' ' + lines[i]
                paras[-1] = previous
        else:
            paras.append(lines[i])
    return paras

# retain newlines and replace them with periods or whitespaces
def softLineBreak1(text):
    lines = []
    tmp = ''
    for ch in text:
        tmp += ch
        if ch == "\n":
            lines.append(tmp)
            tmp = ''

    paras = [lines[0]]
    for i in range(1, len(lines)):
        previous = paras[-1]
        if appendable(previous.strip(), lines[i].strip()):
                if previous[-1] == '\n':
                    previous = previous.replace('\n', ' ')
                previous += lines[i]
                paras[-1] = previous
        else:
            if previous[-1] == '\n':
                previous = previous.replace('\n', '.')
                paras[-1] = previous
            paras.append(lines[i])
    return paras

def softLineBreak2(text):
    lines = []
    tmp = ''
    for ch in text:
        tmp += ch
        if ch == "\n":
            lines.append(tmp)
            tmp = ''

    paras = [lines[0]]
    for i in range(1, len(lines)):
        previous = paras[-1]
        if appendable(previous.strip(), lines[i].strip()):
                if previous[-1] == '\n':
                    previous = previous.replace('\n', ' ')
                previous += lines[i]
                paras[-1] = previous
        else:
            paras.append(lines[i])

    return paras


if __name__ == '__main__':
    [_, inp] = sys.argv

    _, base = os.path.split(inp)
    baseName = base.split('.')[0]

    nextNum = int(re.search('[0-9]+', os.path.basename(inp)).group(0)) + 1

    res = softLineBreak(open(inp).read())

    offset = 0
    for l in res:
        for st, en in sent_tokenizer.span_tokenize(l):
            line = l[st:en].replace('\n', ' ').replace("\t", " ")
            print("\t".join([baseName, str(st + offset), str(en + offset), line.lower()]))
        offset = offset + (len(l) + 1)
