


def init_dict_alphabet(dict_alphabet, dictionary):
    # concept_name may be string or umls_concept
    for concept_id, concept_name in dictionary.items():
        dict_alphabet.add(concept_id)

def get_dict_index(dict_alphabet, concept_id):
    index = dict_alphabet.get_index(concept_id)-1 # pad is 0
    return index

def get_dict_size(dict_alphabet):
    return dict_alphabet.size()-1

def get_dict_name(dict_alphabet, concept_index):
    name = dict_alphabet.get_instance(concept_index+1)
    return name