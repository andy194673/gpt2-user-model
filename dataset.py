import os
import sys
import time
import json
import copy
from itertools import chain
from tqdm import tqdm, trange

import torch
from torch.utils.data import DataLoader, RandomSampler

from utils.argument import get_args
from utils.utils_sgd import load_schema, get_special_tokens, wrap_element, add_str, split_intent

class SGD_Dataset(torch.utils.data.Dataset):
	def __init__(self, args, tokenizer, data_split, generation, data_size):
		assert data_split in ['train', 'dev', 'test', 'demo']
		self.args = args
		self.data_size = data_size
		self.tokenizer = tokenizer
		self.data_split = data_split
		self.generation = generation
		self.n_trimmed = 0

		self.SPECIAL_TOKENS = get_special_tokens()
		self._get_special_token_ids()

		# create examples
		self.examples = []
		for data_name in args.data_list:
			examples = self._create_examples(data_name, data_split)
			self.examples += examples
		print("Total ({}) -> {} examples".format(data_split, len(self.examples)))


	def _get_special_token_ids(self):
		self.bos_id = self.tokenizer.convert_tokens_to_ids(self.SPECIAL_TOKENS["bos_token"])
		self.eos_id = self.tokenizer.convert_tokens_to_ids(self.SPECIAL_TOKENS["eos_token"])
		self.pad_id = self.tokenizer.convert_tokens_to_ids(self.SPECIAL_TOKENS["pad_token"])
		self.sep_id = self.tokenizer.convert_tokens_to_ids(self.SPECIAL_TOKENS["sep_token"])
#		print('SPECIAL TOKEN MAPPING:')
#		print('bos:{} | eos:{} | pad:{} | sep:{}'.format(self.bos_id, self.eos_id, self.pad_id, self.sep_id))

		self.add_special_token_ids = {}
		for token in self.SPECIAL_TOKENS["additional_special_tokens"]:
			self.add_special_token_ids[token] = self.tokenizer.convert_tokens_to_ids(token)

		self.true_token, self.false_token = "_True_", "_False_"
		assert self.true_token in self.SPECIAL_TOKENS["additional_special_tokens"]
		assert self.false_token in self.SPECIAL_TOKENS["additional_special_tokens"]
		'''
		if using BPE (default method, simply call tokenizer(natural sentence)), no need unk_token
		if using convert_tokens_to_ids, check which is correct way to handle oov:
			a) simply use <endoftext> as unk_token (default setup) or
			b) add unk_token into special tokens
		'''

	def _create_examples(self, data_name, data_split):
		data_file = os.path.join(self.args.data_dir, data_name, "{}.json".format(data_split))
		with open(data_file) as f:
			data = json.load(f)

		examples = []
		for dial_id in tqdm(sorted(data.keys())):
			if self.data_size != -1 and len(examples) >= self.data_size:
				break
			dial_meta = data[dial_id]
			context = ""
			for i in range(100):
				example_id = "{}-{}".format(dial_id, i)
				self.example_id = example_id
				if example_id not in dial_meta:
					break

				###### testing #####
#				# SGD
#				if data_split == "test" and dial_id not in ["10_00056", "10_00075"]: # seen, movie domain
#				if data_split == "test" and dial_id not in ["16_00040"]: # seen
#				if data_split == "test" and dial_id not in ["8_00066", "16_00095", "8_00065"]: # unseen
#				if data_split == "test" and dial_id not in ["9_00121", "9_00122"]: # req_alts cases w/i, w/o inform
#					continue
#				# mwoz
#				if data_split == "test" and dial_id not in ["MUL0071.json"]: # test predictions in no offer & no book
#					continue

				# turn info
				goal = dial_meta[example_id]["goal"]
				service = dial_meta[example_id]["service"]
				intent = dial_meta[example_id]["intent"]

				# utterances
				usr_utt = dial_meta[example_id]["utterances"]["usr"]
				sys_utt = dial_meta[example_id]["utterances"]["sys"]

				# actions
				usr_act = dial_meta[example_id]["actions"]["usr"]
				sys_act = dial_meta[example_id]["actions"]["sys"]

				# binary flags
				snt = dial_meta[example_id]["start_new_task"]
				gc = dial_meta[example_id]["goal_change"]
				ra = dial_meta[example_id]["req_alts"]

				# get input ids
				input_seq, input_ids, label_ids, valid_example = self._prepare_input_ids(goal, context, usr_utt, usr_act, sys_utt, sys_act, snt, gc, ra)

				if valid_example:
					assert len(input_ids) < 1024
					dial_meta[example_id]["context"] = context
					examples.append({
						"input_ids": input_ids, # list of ids
						"label_ids": label_ids, # list of ids
						"metadata": dial_meta[example_id],
						"example_id": self.example_id,
						"data_name": data_name
					})

				# collect context
				sys_utt_wrap = wrap_element("SYS", sys_utt)
				usr_utt_wrap = wrap_element("USR", usr_utt)
				context = add_str(context, sys_utt_wrap)
				context = add_str(context, usr_utt_wrap)

		print('Data Stat: {} ({}) -> {} examples ({} examples are trimmed)'.format(data_name, self.data_split, len(examples), self.n_trimmed))
		return examples


	def _prepare_input_ids(self, goal, context, usr_utt, usr_act, sys_utt, sys_act, snt, gc, ra):
		'''
			prepare input sequence ids to GPT2
			template: <CTX> <SYS_UTT> <SYS_ACT> <SNT> <RA> <GC> <GOAL> <USR_ACT> <USR_UTT>
		'''
		goal_wrap = wrap_element("GOAL", goal)
		context_wrap = wrap_element("CTX", context)
		usr_utt_wrap = wrap_element("USR_UTT", usr_utt)
		usr_act_wrap = wrap_element("USR_ACT", usr_act)
		sys_utt_wrap = wrap_element("SYS_UTT", sys_utt)
		sys_act_wrap = wrap_element("SYS_ACT", sys_act)

		snt = self.true_token if snt else self.false_token # `Start New Task` flag
		snt_wrap = wrap_element("SNT", snt)
		gc = self.true_token if gc else self.false_token # `Goal Change` flag
		gc_wrap = wrap_element("GC", gc)
		ra = self.true_token if ra else self.false_token # `Request Alternatives` flag
		ra_wrap = wrap_element("RA", ra)
		if self.args.use_ra_flag:
			flags_wrap = snt_wrap + " " + ra_wrap + " " + gc_wrap
		else:
			flags_wrap = snt_wrap + " " + gc_wrap

		if not self.generation: # supervised
			input_seq = context_wrap + " " + sys_utt_wrap + " " + sys_act_wrap + " " + flags_wrap + " " + \
							goal_wrap + " " + usr_act_wrap + " " + usr_utt_wrap + " " + self.SPECIAL_TOKENS["eos_token"]
			input_ids = self.tokenizer(input_seq)["input_ids"] # convert to ids
			label_ids = self._get_labels(input_ids)
		else: # generation
			input_seq = context_wrap + " " + sys_utt_wrap + " " + sys_act_wrap + " " + flags_wrap + " " + \
							goal_wrap + " " + "<USR_ACT/>" #+ " " + usr_act_wrap + " " + usr_utt_wrap
			input_ids = self.tokenizer(input_seq)["input_ids"] # convert to ids
			label_ids = None

		valid_example = True
		if len(input_ids) > 1023:
			print("{}: {}".format(self.n_trimmed, self.example_id))
			self.n_trimmed += 1
			valid_example = False

		return input_seq, input_ids, label_ids, valid_example


	def _get_labels(self, input_ids):
		for special_token in ["<SYS_ACT/>", "</GC>", "<USR_ACT/>"]:
			special_token_id = self.add_special_token_ids[special_token]
			assert input_ids.count(special_token_id) == 1

		label_ids = [-100] * len(input_ids)

		# sys act signal interval
		start_position = input_ids.index(self.add_special_token_ids["<SYS_ACT/>"])
		end_position   = input_ids.index(self.add_special_token_ids["</GC>"]) + 1
		label_ids[start_position: end_position] = input_ids[start_position: end_position]

		# usr act and utt singal interval 
		start_position = input_ids.index(self.add_special_token_ids["<USR_ACT/>"])
		assert self.eos_id == input_ids[-1]
		label_ids[start_position: ] = input_ids[start_position: ]
		assert len(label_ids) == len(input_ids)
		return label_ids
		

	def _pad(self, sentences, pad_id):
		max_len = max((map(len, sentences)))
		attention_mask = []
		sentences_pad = []
		for sent in sentences:
			pad_len = max_len - len(sent)
			sentences_pad.append( sent + [pad_id]*pad_len )
			attention_mask.append( [1]*len(sent) + [0]*pad_len)
		return sentences_pad, attention_mask


	def __len__(self): # required
		return len(self.examples)


	def __getitem__(self, index): # required
		'''
			index will be ramdomly sampled by the fed sampler, we dont need to worry about index
		'''
		return self.examples[index]


	def collate_fn(self, batch): # optional but useful
		'''
			when collate_fn is given to the torch dataloader, we can do further actions to the batch, e.g., tensor can be formed here
			a batch is formed as a list where each element is a defined data returned by __getitem__, andy
		'''
		input_ids = [example['input_ids'] for example in batch]
		input_ids, attention_mask = self._pad(input_ids, self.pad_id)
		input_ids, attention_mask = torch.tensor(input_ids).long().to(self.args.device), torch.tensor(attention_mask).long().to(self.args.device)

		if not self.generation:
			label_ids = [example['label_ids'] for example in batch]
			label_ids, _ = self._pad(label_ids, -100)
			label_ids = torch.tensor(label_ids).long().to(self.args.device)
		else:
			label_ids = None
		token_type_ids = None

		# store info for scoring
		metadata = [ex["metadata"] for ex in batch]
		example_id = [ex["example_id"] for ex in batch]
		data_name = [ex["data_name"] for ex in batch]
		
		return {"input_ids": input_ids, "attention_mask": attention_mask, "token_type_ids": token_type_ids, "label_ids": label_ids, \
				"metadata": metadata, "example_id": example_id, "data_name": data_name}


if __name__ == '__main__':
	pass
