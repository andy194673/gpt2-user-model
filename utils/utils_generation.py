import sys, json
import torch
from utils.utils_sgd import *

def find_segment(gen, tag):
	assert isinstance(gen, str)
	gen = gen.split()
	try:
		start = gen.index("<{}/>".format(tag)) + 1
		end   = gen.index("</{}>".format(tag))
		segment = " ".join(gen[start: end])
	except:
		print("Missing {} tag in generated sequence".format(tag))
		segment = None
	return segment


def segment_gen(gen, dial_id):
	def _color(_segment):
		if tag == "CTX":
			_segment = _segment.replace(" </USR>", f"{bcolors.ENDC}")
			_segment = _segment.replace(" </SYS>", f"{bcolors.ENDC}")
			_segment = _segment.replace("<USR/> ", f"USR: {bcolors.OKCYAN}")
			_segment = _segment.replace("<SYS/> ", f"SYS: {bcolors.OKBLUE}")
		if tag == "SYS_UTT":
			_segment = f"{bcolors.OKBLUE}" + _segment + f"{bcolors.ENDC}"
		if tag == "USR_UTT":
			_segment = f"{bcolors.OKCYAN}" + _segment + f"{bcolors.ENDC}"
		if tag in ["SYS_ACT", "USR_ACT", "GOAL"]:
			_segment = _segment.replace("<ACT/> ", f"{bcolors.RED}")
			_segment = _segment.replace(" </ACT>", f"{bcolors.ENDC}")
			_segment = _segment.replace("<SLOT/> ", f"{bcolors.YELLOW}")
			_segment = _segment.replace(" </SLOT>", f"{bcolors.ENDC}")
			_segment = _segment.replace("<VALUE/> ", f"{bcolors.GREEN}")
			_segment = _segment.replace(" </VALUE>", f"{bcolors.ENDC}")
		if tag == "GOAL":
			_segment = _segment.replace("<SCENARIO/>", f"<SCENARIO/>{bcolors.UNDERLINE}")
			_segment = _segment.replace("</SCENARIO>", f"{bcolors.ENDC}</SCENARIO>")
			_segment = _segment.replace("<TASK/>", f"<TASK/>{bcolors.UNDERLINE}")
			_segment = _segment.replace("</TASK>", f"{bcolors.ENDC}</TASK>")
#		if tag in ["SNT", "GC"]:
#			segment = segment.replace("<{}/> ".format(tag), "<{}/> *".format(tag))
#			segment = segment.replace(" </{}>".format(tag), "* <{}/>".format(tag))
		return _segment

	assert isinstance(gen, str)
	print("*** Dial_id: {} ***".format(dial_id))
	for tag in ["CTX", "SYS_UTT", "SYS_ACT", "GOAL", "SNT", "RA", "GC", "USR_ACT", "USR_UTT"]:
		segment = find_segment(gen, tag)
		if segment is not None:
			print('{} -> "{}"'.format(tag, _color(segment)))
		else:
			print("Fail to find the segment...")
			print("GEN:", gen)
	print("---"*30)
	input("press any key to continue...")


def save_gen(gen, dial_id, container):
	output = {'raw_generation': gen}
	parsed_generation = {}

	assert isinstance(gen, str)
	for tag in ["CTX", "SYS_UTT", "SYS_ACT", "GOAL", "SNT", "RA", "GC", "USR_ACT", "USR_UTT"]:
		segment = find_segment(gen, tag)
		if segment is not None:
			parsed_generation[tag] = segment
		else:
			print("Fail to parse generation on example {}".format(dial_id))
			parsed_generation[tag] = None

	output['parsed_generation'] = parsed_generation
	container[dial_id] = output


#def decode(args, batch, model, tokenizer):
#	input_ids = batch['input_ids']
#	batch_size, ctx_len = input_ids.size()
#	assert batch_size == 1
#	bos_id, eos_id, pad_id, sep_id = tokenizer.convert_tokens_to_ids(['<BOS>', '<EOS>', '<PAD>', '<SEP>'])
#
#	# output size: (B, T)
#	output = model.generate(input_ids, max_length=(ctx_len+args.dec_max_len), do_sample=False, temperature=args.temperature, use_cache=True, num_beams=args.num_beams, bos_token_id=bos_id, eos_token_id=eos_id, pad_token_id=pad_id, early_stopping=True)
#	gen = tokenizer.decode(output[0]) # include context fed into model
#	segment_gen(gen, batch["example_id"][0])
#	return [gen]


def prepare_input_ids(args: object, tokenizer: object, data: object, start_token: object) -> object:
	assert start_token in ["<SYS_ACT/>", "<USR_ACT/>"]
	input_seq = ""
	for key in ["CTX", "SYS_UTT", "SYS_ACT", "SNT", "RA", "GC", "GOAL"]: # fixed order, consistent between training and inference
		if key not in data:
			continue
		wrap = wrap_element(key, data[key])
		input_seq = add_str(input_seq, wrap)

	input_seq = add_str(input_seq, start_token)

	input_ids = tokenizer(input_seq)["input_ids"] # convert to ids
	input_ids = torch.tensor([input_ids]).long().to(args.device)
	return input_ids


def decode_e2e(args, batch, model, tokenizer, user_goal=None, prev_usr_act=None, collector=None):
	""" decode with predicted sys act, goal can be random or from the corpus """
	assert len(batch["metadata"]) == 1
	context = batch["metadata"][0]["context"]
	sys_utt = batch["metadata"][0]["utterances"]["sys"]
	bos_id, _, pad_id, sep_id = tokenizer.convert_tokens_to_ids(['<BOS>', '<EOS>', '<PAD>', '<SEP>'])

	# first forward pass
	data = {"CTX": context, "SYS_UTT": sys_utt}
	start_token, end_token = "<SYS_ACT/>", "</GC>"
	input_ids = prepare_input_ids(args, tokenizer, data, start_token)
	eos_id = tokenizer.convert_tokens_to_ids(end_token)
	output = model.generate(input_ids, max_length=args.dec_max_len, do_sample=False, temperature=args.temperature, use_cache=True,
							num_beams=args.num_beams, bos_token_id=bos_id, eos_token_id=eos_id, pad_token_id=pad_id, early_stopping=True)
	gen = tokenizer.decode(output[0]) # include context fed into model

	# parse the first pass prediction
	for key in ["SYS_ACT", "SNT", "GC", "RA"]:
		value = find_segment(gen, key)
		data[key] = value
	# print("***** First run generation *****")
	# print("SYS_ACT -> {}".format(data["SYS_ACT"]))
	# print("FLAGS -> SNT: {}, GC: {}, RA: {} *****".format(data["SNT"], data["GC"], data["RA"]))
	# print("********************************")

	# prepare goal
	if user_goal is None: # use ground truth goal from corpus
		data["GOAL"] = batch["metadata"][0]['goal']
	else:
		goal = user_goal.prepare_turn_goal(prev_usr_act, data["SYS_ACT"], data["SNT"], data["GC"], data["RA"])
		data["GOAL"] = goal

	# second forward pass
	start_token, end_token = "<USR_ACT/>", "<EOS>"
	input_ids = prepare_input_ids(args, tokenizer, data, start_token)
	eos_id = tokenizer.convert_tokens_to_ids(end_token)
	output = model.generate(input_ids, max_length=args.dec_max_len, do_sample=False, temperature=args.temperature, use_cache=True,
							num_beams=args.num_beams, bos_token_id=bos_id, eos_token_id=eos_id, pad_token_id=pad_id, early_stopping=True)
	gen = tokenizer.decode(output[0]) # include context fed into model
	if args.eye_browse_output:
		segment_gen(gen, batch["example_id"][0])
	else:
		save_gen(gen, batch["example_id"][0], collector)
	return [gen]
