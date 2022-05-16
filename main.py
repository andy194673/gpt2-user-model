import os
import sys
import json
import time
import random
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
import numpy as np
from transformers import (
	AdamW,
	AutoConfig,
	AutoTokenizer,
	GPT2LMHeadModel,
	get_linear_schedule_with_warmup,
)

from utils.argument import get_args
from utils.utils_sgd import get_special_tokens
from utils.utils_generation import *
#from interact import interact
from dataset import SGD_Dataset


def print_loss(epoch, data_type, LOSS, t0):
	print('Epoch: {} | {} loss: {:.3f} | time: {:.1f}'.format(epoch, data_type, LOSS, time.time()-t0))

def print_score(epoch, data_type, res, t0):
	print('Epoch: {} | {}: joint_acc: {:.2f}%, slot_acc: {:.2f}% | time: {:.1f}'.format(epoch, data_type, res['avg_joint_acc'], res['avg_slot_acc'], time.time()-t0))


def run_one_epoch(data_type, dataloader, trainer, epoch, run_type, collector=None):
	t0 = time.time()
	assert data_type in ['dev', 'test']
	assert run_type in ['teacher_force', 'generation']
	model, optimizer, scheduler, tokenizer = trainer

	LOSS = 0
	result = {'slot_acc': [], 'joint_acc': []}
	mention_match = 0
	coref_lines = []
	iterator = enumerate(tqdm(dataloader, desc="Epoch {} {}".format(epoch, run_type), disable=args.disable_display))
	for step, batch in iterator:
		if run_type == 'teacher_force':
			loss, logits, _ = model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'], token_type_ids=batch['token_type_ids'], labels=batch['label_ids'])
			LOSS += loss
		else:
			decode_e2e(args, batch, model, tokenizer, collector=collector)

	# print log
	if run_type == 'teacher_force':
		LOSS /= (step+1)
		print_loss(epoch, data_type, LOSS, t0)
		return LOSS
	else: # generation
		# TODO: add evaluation code here
		return None


def set_dataloader(args, tokenizer, data_type, run_type, data_size=-1):
	dataset = SGD_Dataset(args, tokenizer, data_type, run_type=='generation', data_size)
#	sys.exit(1)
	if data_type == 'train':
		sampler = RandomSampler(dataset) #if args.local_rank == -1 else DistributedSampler(train_dataset)
	else:
		sampler = SequentialSampler(dataset)

	dataloader = DataLoader(
		dataset,
		sampler=sampler,
		batch_size=args.train_batch_size if data_type == 'train' else args.eval_batch_size,
		collate_fn=dataset.collate_fn
	)
	return dataloader


def train(args, tokenizer, model):
	# load data
	train_dataloader = set_dataloader(args, tokenizer, 'train', 'teacher_force', data_size=args.train_size)
	dev_dataloader = set_dataloader(args, tokenizer, 'dev', 'teacher_force', data_size=args.eval_size)

	optimizer = AdamW(model.parameters(), lr=args.learning_rate, eps=args.adam_epsilon)
	if args.use_scheduler:
		t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.max_epoch
		scheduler = get_linear_schedule_with_warmup(
			optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total
		)
	else:
		scheduler = None
	trainer = (model, optimizer, scheduler, tokenizer)

	print('Do evaluation before training!')
	model.eval()
	with torch.no_grad():
		_ = run_one_epoch('dev', dev_dataloader, trainer, -1, 'teacher_force')

	print('Start training!\n{}'.format('***'*30))
	eval_step = args.eval_interval // args.train_batch_size
	best_score = -100
	global_step = 0
	no_improve_count = 0
	for epoch in range(args.max_epoch):
		# initialize for each epoch training
		t0 = time.time()
		model.train()
		model.zero_grad()
		LOSS = 0
		iterator = enumerate(tqdm(train_dataloader, desc="Epoch {}".format(epoch), disable=args.disable_display))
		for local_step, batch in iterator:
			loss, logits, _ = model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'], token_type_ids=batch['token_type_ids'], labels=batch['label_ids'])
			LOSS += loss.item()
			global_step += 1

			# update model
			if loss.item() != 0:
				loss = loss / args.gradient_accumulation_steps
				loss.backward()

			if global_step % args.gradient_accumulation_steps == 0:
				norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
				optimizer.step()
				if args.use_scheduler:
					scheduler.step()
				optimizer.zero_grad()

			# evaluate model
			if global_step % eval_step == 0: 
				model.eval()
				with torch.no_grad():
					loss = run_one_epoch('dev', dev_dataloader, trainer, epoch, 'teacher_force')
					score = -loss # dev loss as criterion for early training
				model.train()

				save_checkpoint(args, tokenizer, model, global_step*args.train_batch_size)
				if score > best_score:
					best_score = score
					print('Best score: {:.2f}'.format(best_score))
					no_improve_count = 0
				else:
					no_improve_count += 1

				# early stop
				if no_improve_count == args.no_improve_max:
					print('Early stop!')
					return

		LOSS /= (local_step+1)
		print_loss(epoch, 'train', LOSS, t0)
		print('***'*30)


def test(args, tokenizer, model):
	# load data
	test_gen_dataloader = set_dataloader(args, tokenizer, 'test', 'generation')

	trainer = (model, None, None, tokenizer)
	model.eval()
	collector = {'decode-dev': {}, 'decode-test': {}}
	with torch.no_grad():
#		# evaluate on dev
#		_ = run_one_epoch('dev', dev_dataloader, trainer, 'Eval', 'teacher_force')

#		# generate on dev
#		res_dev = run_one_epoch('dev', dev_gen_dataloader, trainer, 'Dev', 'generation', collector=collector['decode-dev'])
#		collector['result-dev'] = res_dev
#		print_qr_result(res_dev['qr'], 'dev')

		# generate on test
		res_test = run_one_epoch('test', test_gen_dataloader, trainer, 'Test', 'generation', collector=collector['decode-test'])
		collector['result-test'] = res_test

	out_file = args.decode_file
	with open(out_file, 'w') as f:
		json.dump(collector, f, indent=4, sort_keys=True)
	print('Decode file is saved at {}'.format(out_file))
	print('Done decoding!')


def save_checkpoint(args, tokenizer, model, step):
	save_path = args.checkpoint + '_step' + str(step)
	print('Save model in {}!'.format(save_path))
	tokenizer.save_pretrained(save_path)
	model.save_pretrained(save_path)
	

def load_checkpoint(args):
	save_path = args.checkpoint #+ '_step' + str(args.step)
	print('Load model, tokenizer from {}'.format(save_path))
	tokenizer = AutoTokenizer.from_pretrained(save_path)
	model = GPT2LMHeadModel.from_pretrained(save_path)
	model.to(args.device)
	return tokenizer, model


def load_pretrained_model(args):
	save_path = args.pre_checkpoint
	print('Load model, tokenizer from {}'.format(save_path))
	tokenizer = AutoTokenizer.from_pretrained(save_path)
	model = GPT2LMHeadModel.from_pretrained(save_path)
	model.to(args.device)
	return tokenizer, model


def set_model(args, SPECIAL_TOKENS):
	''' initiate config, tokenizer and model '''
	# add special tokens into tokenizer
	config = AutoConfig.from_pretrained(args.model_name_or_path)
	tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
	tokenizer.add_special_tokens(SPECIAL_TOKENS)
	model = GPT2LMHeadModel.from_pretrained(args.model_name_or_path, config=config) # GPT2LMHeadModel
	model.resize_token_embeddings(len(tokenizer))
	model.to(args.device)
	print("Done setting model")
	return config, tokenizer, model
	

def set_seed(args):
	''' for reproduction '''
	random.seed(args.seed)
	np.random.seed(args.seed)
	torch.manual_seed(args.seed)
	torch.cuda.manual_seed(args.seed)
	torch.cuda.manual_seed_all(args.seed)
	torch.backends.cudnn.deterministic = True
	torch.backends.cudnn.enabled = False
	torch.backends.cudnn.benchmark = False


if __name__ == '__main__':
	# Load arguments
	args = get_args()

	# Set seed, device
	set_seed(args)
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	args.device = device

	# Load special tokens 
	SPECIAL_TOKENS = get_special_tokens()

	if args.mode == 'training':
		config, tokenizer, model = set_model(args, SPECIAL_TOKENS)
		train(args, tokenizer, model)

	elif args.mode == 'finetune':
		tokenizer, model = load_pretrained_model(args)
		train(args, tokenizer, model)

	elif args.mode == 'testing':
		tokenizer, model = load_checkpoint(args)
		test(args, tokenizer, model)

#	elif args.mode == 'interact':
#		tokenizer, model = load_checkpoint(args)
#		interact(args, tokenizer, model)

	else:
		sys.exit(1)
