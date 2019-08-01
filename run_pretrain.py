import random
import logging
import numpy as np
import torch

from options import opt
from mimic_utils import write_discharge_summaries
from metamap_utils import apply_metamap_to, merge_text_and_metamap
from prepare_instance import prepare_instance
from pretrain import pretrain

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


    if opt.whattodo == 'prepare_text':
        # from mimic to raw text
        write_discharge_summaries(opt.mimic3_dir, opt.text_dir)
        # use metamap to annotate text
        apply_metamap_to(opt.text_dir, opt.metamap_dir, opt.metamap, opt.metamap_process)
        # merge raw text and metamap
        merge_text_and_metamap(opt.text_dir, opt.metamap_dir, opt.merged_file)

    elif opt.whattodo == 'prepare_instance':
        prepare_instance(opt)
    elif opt.whattodo == 'train':
        pretrain(opt)
    else:
        raise RuntimeError("wrong whattodo {}".format(opt.whattodo))

    logging.info('done ........')