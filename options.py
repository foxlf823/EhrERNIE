import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-whattodo', type=str, default="train", help='run_pretrain (prepare_text,prepare_instance,train)')
parser.add_argument('-train', type=str, default='', help="train data path")
parser.add_argument('-test', type=str, default='', help="test data path")

parser.add_argument('-verbose', action='store_true', default=False, help="output debug info if true")
parser.add_argument('-random_seed', type=int, default=1, help="random initial seed if 0")
parser.add_argument('-max_seq_length', type=int, default=256, help="The maximum total input sequence length after WordPiece tokenization. ")

parser.add_argument('-bert_dir', type=str, help="bert directory that includes bert_config, vocab and model")
parser.add_argument('-do_lower_case', action='store_true', default=False, help="used with BertTokenizer")
parser.add_argument('-save', type=str, default='./save', help="output directory for training")
parser.add_argument('-predict', type=str, default='./predict', help="output directory for test")

parser.add_argument('-batch_size', type=int, default=16)
parser.add_argument('-lr', type=float, default=0.00001, help="learning rate")

parser.add_argument('-gpu', type=int, default=-1, help="if gpu<0, use cpu, otherwise use the specific gpu")
parser.add_argument('-multi_gpu', action='store_true', default=False)

parser.add_argument('-iter', type=int, default=10, help="max iteration")
parser.add_argument('-patience', type=int, default=10, help="if the result doesn't rise after several iteration, training will stop")

parser.add_argument("-warmup_proportion", default=0.1, type=float)
parser.add_argument('-gradient_accumulation_steps',
                    type=int,
                    default=1,
                    help="Number of updates steps to accumulate before performing a backward/update pass.")

parser.add_argument("-short_seq_prob", type=float, default=0.1,
                    help="Probability of making a short sentence as a training example")
parser.add_argument("-masked_lm_prob", type=float, default=0.15,
                    help="Probability of masking each token for the LM task")
parser.add_argument("-max_predictions_per_seq", type=int, default=20,
                    help="Maximum number of tokens to mask in each sequence")

# for pretrain
parser.add_argument('-mimic3_dir', type=str, default='')
parser.add_argument('-text_dir', type=str, default='')
parser.add_argument('-metamap', type=str, default='')
parser.add_argument('-metamap_dir', type=str, default='')
parser.add_argument('-merged_file', type=str, default='')
parser.add_argument('-metamap_process', type=int, default=1)
parser.add_argument('-instance_dir', type=str, default='')

# for norm
parser.add_argument('-norm_dict', type=str, default='')



opt = parser.parse_args()

