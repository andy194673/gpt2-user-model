import argparse
import re
import sys

def str2bool(v):
	if v.lower() in ('yes', 'true', 't', 'y', '1'):
		return True
	elif v.lower() in ('no', 'false', 'f', 'n', '0'):
		return False
	else:   
		raise argparse.ArgumentTypeError('Boolean value expected.')


def verify_args(args):
	# datasets
	assert isinstance(args.data_list, list) and len(args.data_list) > 0
	for data_name in args.data_list:
		assert data_name in ["sgd", "multiwoz"]

	# mode
	assert args.mode in ["training", "finetune", "testing", "interact"]
	if args.mode == "finetune":
		assert args.pre_checkpoint != ""


def get_args():
	parser = argparse.ArgumentParser(description='')
	# data
	parser.add_argument('--data_dir', type=str, default="proc_data", help="Directory of processed datasets")
	parser.add_argument('--data_list', type=str, nargs='+', default="", help="Datasets involved, split by space, e.g., `sgd multiwoz`")

	# design control
	parser.add_argument('--use_ra_flag', type=str2bool, default=True, help="Whether to use `request_alternatives` flag")
	
	# training
	parser.add_argument('--mode', type=str, required=True, help='')
	parser.add_argument('--seed', type=int, default=1122)
	parser.add_argument('--model_name', type=str, required=True, help='Unique name, e.g., job id')
	parser.add_argument('--model_name_or_path', type=str, default='gpt2')
	parser.add_argument('--train_batch_size', type=int, default=4, help='Batch size of training per gpu')
	parser.add_argument('--eval_batch_size', type=int, default=1, help='Batch size of evaluation per gpu') # TODO: make decoding parallel
	parser.add_argument('--gradient_accumulation_steps', type=int, default=4)
	parser.add_argument('--learning_rate', type=float, default=6.25e-5) # tune
	parser.add_argument('--adam_epsilon', type=float, default=1e-12)
	parser.add_argument('--max_grad_norm', type=float, default=1.0)
	parser.add_argument('--max_epoch', type=int, default=20)
	parser.add_argument('--fp16', type=str2bool, default=False, help='Whether to use float16')
	parser.add_argument('--use_scheduler', type=str2bool, default=True, help='Whether to use lr scheduler')
	parser.add_argument('--warmup_steps', type=int, default=0)
	parser.add_argument('--checkpoint', type=str, default='', required=True, help='Path of your trained model')
	parser.add_argument('--pre_checkpoint', type=str, default='', help='Path of the pretrained model used for finetuning')
	parser.add_argument('--train_size', type=int, default=-1, help='How many examples used for training. -1 means all data')
	parser.add_argument('--eval_size', type=int, default=-1, help='How many examples used for evaluation. -1 means all data')
	parser.add_argument('--eval_interval', type=int, default=1000, help='During training, how frequent to evaluate the model in terms of training examples')
	parser.add_argument('--no_improve_max', type=int, default=100, help='The max tolerance for model not improving')
	parser.add_argument('--eps', type=float, default=1e-12)
	parser.add_argument('--disable_display', type=str2bool, default=False, help='display progress bar')

	# decoding
#	parser.add_argument('--step', type=int, default=-1) # load model trained at which specific step
	parser.add_argument('--dec_max_len', type=int, default=2000) # we use early stop to stop generation when hits <EOS>
	parser.add_argument('--num_beams', type=int, default=1)
	parser.add_argument('--temperature', type=float, default=1.0)
#	parser.add_argument('--top_k', type=int, default=0)
#	parser.add_argument('--top_p', type=int, default=0)
	parser.add_argument('--decode_file', type=str, default='')
	parser.add_argument('--eye_browse_output', type=str2bool, default=False, help='Whether to eye browse decoded results')

	# ddp
	parser.add_argument('--local_rank', type=int, default=-1, help='Local rank for distributed training (-1: not distributed)')

	args = parser.parse_args()
	verify_args(args)
	print(args)
	return args
