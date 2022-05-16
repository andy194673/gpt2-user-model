import sys, copy, json, os, random
import numpy as np
import torch
from typing import List
from omegaconf import OmegaConf

from utils import find_segment, wrap_element, add_str, load_schema, segment_gen
import transformers
from transformers import AutoTokenizer, GPT2LMHeadModel

class DummyPolicy:
	def init_session(self, ini_goal):  # noqa
		self.goal = ini_goal  # noqa

	def get_goal(self) -> dict:
		"""Returns current user goal.

		Notes
		-----
		``hasattr`` user works around the fact that ``convlab2`` initialises the dialogue session
		before we can explicitly pass the goal to the user model.
		"""
		if hasattr(self.goal, "domain_goals"):
			return self.goal.domain_goals
		# return {}
		return self.goal # for consistency


def generation_func(model, input_ids, eos_id, dec_max_len):
	''' Generation method using greedy search for Transformer v2.x '''
	def _extend_mask(mask):
		mask = torch.cat([mask, mask.new_ones((mask.shape[0], 1))], dim=-1)
		return mask

	# input_ids, attention_mask, token_type_ids =  batch['input_ids'], batch['attention_mask'], batch['token_type_ids']
	batch_size = input_ids.size(0)
	attention_mask = torch.ones_like(input_ids)
	past = None
	finish_sent = [False for _ in range(batch_size)]
	for i in range(dec_max_len):
		logits, past = model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=None)

		# logits: (B, T, V), T=1 when past is passed
		next_token_logits = logits[:, -1, :]
		next_token = torch.argmax(next_token_logits, dim=-1)
		input_ids = torch.cat([input_ids, next_token.unsqueeze(-1)], dim=-1)
		attention_mask = _extend_mask(attention_mask)

		for bs_idx, token_id in enumerate(next_token):
			if finish_sent[bs_idx] is False and token_id.item() == eos_id: # first produce <eos>
				finish_sent[bs_idx] = True
		if sum(finish_sent) == batch_size:
			break
	return input_ids


class NeuralAgent: # crazyusermodel

	def __init__(self, name: str, model_path: str, model_config_path: str):
		"""User Simulator
		Description
		---------
			A user model that is able to chat with the task-oriented dialogue system in an end-to-end manner

        Parameters
        ----------
        name
           Should indicate the role played by the agent. It should be always user
		"""

		if name != "user":
			raise ValueError(f"Expected name 'user' but got {name} instead.")

		# load necessities
		self.set_device()
		self.config = OmegaConf.load(model_config_path)

		# get schema, which is dependent to dataset, only for providing task description here
		self.service2meta, self.schema_intents, self.schema_slots = load_schema(self.config["schema_path"])
#		self.load_checkpoint_and_tokenizer(self.config["model"]["path"])
		self.load_checkpoint_and_tokenizer(model_path)
		self.load_materials()

		self.context = []
		self.current_goal = {}
		self.behaviour_params = {}
		self.input_action = []  # type: list[list[str]]
		self.output_action = []  # type: list[list[str]]

		# for compatibility with convlab2 evaluator
		self.policy = DummyPolicy()

		''' for reproduction '''
		seed = 1130
		random.seed(seed)
		np.random.seed(seed)
		torch.manual_seed(seed)
		torch.cuda.manual_seed(seed)
		torch.cuda.manual_seed_all(seed)
		torch.backends.cudnn.deterministic = True
		torch.backends.cudnn.enabled = False
		torch.backends.cudnn.benchmark = False


	def load_checkpoint_and_tokenizer(self, checkpoint_path: str) -> None:
		"""Load model checkpoint with the model tokenizer, only for GPT2 for now"""
		print('Load model, tokenizer from {}'.format(checkpoint_path))
		self.tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
		self.model = GPT2LMHeadModel.from_pretrained(checkpoint_path)
		self.model.to(self.device)


	def set_device(self) -> None:
		"""Set device to GPU/CPU"""
		self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


	def load_materials(self):
		"""Load useful materials used in generation"""
		# model attributes
		'''
		finish_inform
			how strict to finish an informable slot in goal: "strict" or "loose"
			if "strict": the attribute are finished (removed from goal) only if both slot and value are produced in act
			if "loose": the attribute are finished if the slot is produced
		'''
		self.finish_inform = self.config.model.goal_update.finish_inform # controls how strict to eliminate informed slots in goal
		assert self.finish_inform in ["strict", "loose"]

		# constants
		self.bos_id, _, self.pad_id, self.sep_id = self.tokenizer.convert_tokens_to_ids(['<BOS>', '<EOS>', '<PAD>', '<SEP>'])
		self.bin_flags = {"true": "_True_", "false": "_False_"}

		self.supported_services = ["train", "attraction", "hotel", "restaurant", "taxi"]

		self.slot_types = {"search": "search", "book": "book"}
		# important to change the corresponding act str name when using different tokenization methods,
		# as they are used to control the user behaviours
		self.const_act_str = {"inform": "inform", "request": "request", "fail_search": "no offer", "fail_book": "no book"}


	def prepare_input_ids(self, data: dict, start_token: str) -> str:
		assert start_token in ["<SYS_ACT/>", "<USR_ACT/>"]
		input_seq = ""
		for key in ["CTX", "SYS_UTT", "SYS_ACT", "SNT", "RA", "GC", "GOAL"]:  # fixed order, consistent between training and inference
			if key not in data:
				continue
			wrap = wrap_element(key, data[key])
			input_seq = add_str(input_seq, wrap)

		input_seq = add_str(input_seq, start_token)
		if transformers.__version__.startswith("2."): # compatible with transformers v2.x used in convlab2
			input_ids = self.tokenizer.encode(input_seq)
		else:
			input_ids = self.tokenizer(input_seq)["input_ids"]  # convert to ids
		input_ids = torch.tensor([input_ids]).long().to(self.device)
		return input_ids


	def update_internal_data(self, data: dict) -> None:
		"""Maintain context and user act in the format of generation string for the next turn generation"""
		# update context
		sys_utt_wrap = wrap_element("SYS", data["SYS_UTT"]) # e.g., <SYS/> Which area would you prefer? </SYS>
		usr_utt_wrap = wrap_element("USR", data["USR_UTT"]) # e.g., <USR/> I want to be in the centre. </USR>
		self._context_str = add_str(self._context_str, sys_utt_wrap)
		self._context_str = add_str(self._context_str, usr_utt_wrap)

		# update prev usr act
		self._prev_usr_act = data["USR_ACT"] # e.g., <ACT/> inform </ACT> <SLOT/> area </SLOT> <VALUE/> centre </VALUE>


	def run_inference_once(self, input_ids: torch.tensor, eos_id: int) -> List:
		if transformers.__version__.startswith("2."): # compatible with transformers v2.x used in convlab2
			output = generation_func(self.model, input_ids, eos_id, self.config.decode.dec_max_len)
		else:
			output = self.model.generate(
				input_ids,
				max_length=self.config.decode.dec_max_len,
				do_sample=self.config.decode.do_sample,
				early_stopping=True,
				temperature=self.config.decode.temperature,
				use_cache=True,
				num_beams=self.config.decode.num_beams,
				bos_token_id=self.bos_id,
				eos_token_id=eos_id,
				pad_token_id=self.pad_id
			)
		return output


	def generate_whole_sequence(self, sys_utt: str) -> tuple:
		##### first forward pass: generate NLU output and three special flags #####
		data = {"CTX": self._context_str, "SYS_UTT": sys_utt}
		start_token, end_token = "<SYS_ACT/>", "</GC>"
		input_ids = self.prepare_input_ids(data, start_token)
		eos_id = self.tokenizer.convert_tokens_to_ids(end_token)
		output = self.run_inference_once(input_ids, eos_id)
		generation = self.tokenizer.decode(output[0]) # decode back to str, including the fed context

		# parse first pass prediction
		for key in ["SYS_ACT", "SNT", "GC", "RA"]:
			value = find_segment(generation, key)
			data[key] = value

		# update dynamic goal
		print("SYS ACT ->", data["SYS_ACT"])
		goal = self.prepare_turn_goal(self._prev_usr_act, data["SYS_ACT"], data["SNT"], data["GC"], data["RA"])
		data["GOAL"] = goal

		##### second forward pass: generate dialogue act and NLG output #####
		start_token, end_token = "<USR_ACT/>", "<EOS>"
		input_ids = self.prepare_input_ids(data, start_token)
		eos_id = self.tokenizer.convert_tokens_to_ids(end_token)
		output = self.run_inference_once(input_ids, eos_id)
		generation = self.tokenizer.decode(output[0])  # decode back to str, including the fed context

		# parse second pass prediction
		for key in ["USR_ACT", "USR_UTT"]:
			value = find_segment(generation, key)
			data[key] = value
		return data, generation


	def _format_complete_goal(self, input_goal: dict) -> dict:
		"""Format the internal goal representation given a goal

		:param input_goal: a goal that the user has in mind
					 either from the corpus or sampled randomly in a valid way (e.g., correct slot names)
		:returns: complete_goal: an internal representation of the given goal, a dict with the keys "intents", "constraints"
			intents: list[str], list of intents in the dialogue, aka scenario
			constraints: dict, intent as key, in the following format
				dict(intent: intent_constraints)
				intent_constraints: {"informable": dict(slot: value_list), "requestable": slot_set}
				each slot has a value list in case of failure of searching
		"""
		# TODO: make the order of services more flexible (how does convlab2 decide the service order?)
		constraints = dict()
		intents = []
		self.n_max_value = {self.slot_types["book"]: 0, self.slot_types["search"]: 0} # record the max length of value list of a slot
		for service in self.supported_services:
			if service not in input_goal:
				continue

			# record intent list (scenario), order matters
			intent = self._map_service_to_intent(service)
			assert intent not in intents and intent not in constraints
			intents.append(intent)
			constraints[intent] = {"informable": dict(), "requestable": set()}

			# collect informable slots
			assert "info" in input_goal[service] # info has to exist
			for key in ["fail_info", "info", "fail_book", "book"]: # order matters
				# assert key in input_goal[service]
				if key not in input_goal[service]:
					continue
				for slot, value in input_goal[service][key].items():
					self._add_info(constraints[intent]["informable"], slot, value)

			# collect requestable slots
			key = "reqt"
			# assert key in input_goal[service]
			# for slot in input_goal[service][key]:
			if key in input_goal[service]:
				for slot in input_goal[service][key].keys():
					self._add_reqt(constraints[intent]["requestable"], slot)

		complete_goal = {"intents": intents, "constraints": constraints}
		return complete_goal


	def _init_user_status(self) -> dict:
		"""Initialise user status with intent and constraint
			intent_idx: int, the index of current intent
			constraint_idx: dict, intent as key, value is the constraint index used to record which value is used
				in the slot value list
		:return:
		"""
		intent_idx = 0 # -1
		# constraint_idx = {intent: 0 for intent in self.complete_goal["intents"]}
		constraint_idx = {intent: {self.slot_types["search"]: 0, self.slot_types["book"]: 0} for intent in self.complete_goal["intents"]}
		# TODO: entity provide records, one of the criteria to move to the next intents
		entity_provided = {intent: False for intent in self.complete_goal["intents"]}
		return {"intent_idx": intent_idx, "constraint_idx": constraint_idx,
				"dialogue_terminate": False, "entity_provided": entity_provided}


	def _get_scenario_str(self) -> None:
		"""Get a scenario str from a intent list

		Description
			convert a list of intents, aka scenario, into string with special marks
			the scenario is determined at the start of dialogue and static during interaction
		"""
		intents = self.complete_goal["intents"]
		_str = [wrap_element("INTENT", intent) for intent in intents]
		_str = " ".join(_str)
		self.scenario_str = wrap_element("SCENARIO", _str)


	def _prepare_current_constraints(self, involved_intents: List[str], involved_slot_types: List[str], if_reset_reqt: bool) -> None:
		"""Prepare the current constraints, copied the specified content from the complete goal

		the current constraints is used as condition in the model generation
		its content comes from the "constraints" in "complete goal",
		but the current constraints only allows one value for a slot at a time
		the value is chosen from the value list by the "constraint_idx" in user status

		:param involved_intents: list[str], intent list
		:return:
			current_constraints: dict, similar format as constraints in the complete goal,
			but a slot has only one value, e.g.,
				dict(intent: intent_constraints)
				intent_constraints: {"informable": dict(slot: value), "requestable": slot_set}
		"""
		# iterate the involved intents
		for intent in involved_intents:
			constraints = {"informable": dict(), "requestable": set()}
			# informable slots value pairs
			for slot, value_list in self.complete_goal["constraints"][intent]["informable"].items():
				slot_type = self._get_slot_type(slot)
				if slot_type not in involved_slot_types:
					continue
				value_idx = self.user_status["constraint_idx"][intent][slot_type]
				if value_idx < len(value_list):
					value = value_list[value_idx]
					constraints["informable"][slot] = value

			# requestable
			if if_reset_reqt:
				constraints["requestable"] = copy.deepcopy(self.complete_goal["constraints"][intent]["requestable"])
			else:
				constraints["requestable"] = copy.deepcopy(self.current_constraints[intent]["requestable"])

			# overwrite intent constraints
			self.current_constraints[intent] = constraints


	@staticmethod
	def _map_intent_to_service(intent: str) -> str:
		# TODO: make it not dataset dependent?
		"""map an intent into a service, multiwoz only"""
		return intent.split()[1]


	@staticmethod
	def _map_service_to_intent(service: str) -> str:
		# TODO: make it not dataset dependent?
		"""map a service into an intent, multiwoz only"""
		return f"find {service}"


	def _get_slot_type(self, slot: str) -> str:
		"""return search or book type of a slot"""
		slot_type = "book" if "book" in slot else "search"
		assert slot_type in self.slot_types.keys()
		return slot_type


	def _get_goal_str(self, intent: str) -> str:
		"""prepare the proper goal sequence, same as used in training"""
		goal_str = ""
		# dialogue scenario
		goal_str = add_str(goal_str, self.scenario_str)

		# current task
		goal_str = add_str(goal_str, wrap_element("TASK", intent))

		# task description
		service = self._map_intent_to_service(intent)
		description = self.service2meta[service]["intents"][intent]["description"]
		goal_str = add_str(goal_str, wrap_element("DESC", description))

		# intent_constraints = self.dynamic_constraints[intent]
		intent_constraints = self.current_constraints[intent]
		# informable slots
		info_str = ""
		#		for slot, value in intent_constraints["informable"].items():
		for slot in sorted(intent_constraints["informable"].keys()):  # sort by slot
			value = intent_constraints["informable"][slot]
			info_str = add_str(info_str, wrap_element("SLOT", slot))
			info_str = add_str(info_str, wrap_element("VALUE", value))
		goal_str = add_str(goal_str, wrap_element("INFORM", info_str))

		# requestable slots
		req_str = ""
		for slot in sorted(list(intent_constraints["requestable"])):
			req_str = add_str(req_str, wrap_element("SLOT", slot))
		goal_str = add_str(goal_str, wrap_element("REQUEST", req_str))
		return goal_str.strip()


	def _start_new_intent(self, SNT_flag: str) -> bool:
		"""decide whether to start a new intent"""
		# SNT (start new task) is predicted as on
		assert SNT_flag in list(self.bin_flags.values())
		# intent = self.intents[self.intent_idx]
		intent = self.complete_goal["intents"][self.user_status["intent_idx"]]

		# TODO: need at least an entity provided (not really sure...
		# if not self.intent_entity_provided[intent]: # no entities provided in the intent yet
		# 	return False

		# TODO: think about the priority of SNT prediction. It's should be less prioritised than the number of left constraints.
		# if SNT_flag == self.bin_flags["true"]: # model prediction in first turn is true
		# 	return True

		# current intent has empty constraints
		if len(self.current_constraints[intent]["informable"]) == 0 and len(self.current_constraints[intent]["requestable"]) == 0:
			return True
		return False


	def _check_entity_provided(self, sys_act, intent):
		"""Check if an entity provided in system response (act)"""
		assert intent in ["find restaurant", "find hotel", "find attraction", "find train", "find taxi"]
		if intent in ["find restaurant", "find hotel", "find attraction"]:
			if "<SLOT/> name </SLOT>" in sys_act:
				self.intent_entity_provided[intent] = True
		elif intent == "find train":
			if "<SLOT/> train id </SLOT>" in sys_act:
				self.intent_entity_provided[intent] = True
		else:  # taxi
			if "<SLOT/> type </SLOT>" in sys_act:
				self.intent_entity_provided[intent] = True


	def _activate_dialogue_terminate(self) -> None:
		"""Turn on the user status about dialogue termination"""
		self.user_status["dialogue_terminate"] = True


	def prepare_turn_goal(self, prev_usr_act: str, sys_act: str, SNT_flag: str, GC_flag: str, RA_flag: str) -> str:
		"""prepare the goal sequence for the current turn"""
		# TODO: more detailed instruction here
		intent = self.complete_goal["intents"][self.user_status["intent_idx"]]

		# TODO: check if at least one entity is provided in system act
		# First thing to do, check if the system provides an entity
		# self._check_entity_provided(sys_act, intent)

		# update goal first then check if moves to next intent (task)
		self._update_current_constraints(intent, "usr", prev_usr_act, sys_act)
		self._update_current_constraints(intent, "sys", prev_usr_act, sys_act) # impact of sys_act overwrites that of usr_act

		# check if new intent starts
		if self._start_new_intent(SNT_flag):
			self.user_status["intent_idx"] += 1
			if self.user_status["intent_idx"] < len(self.complete_goal["intents"]):
				intent = self.complete_goal["intents"][self.user_status["intent_idx"]]
			else:
				self._activate_dialogue_terminate()
		# TODO: request alternative by setting <RA> for sgd
		# TODO: sample new goal if goal change <GC> for sgd
		goal_str = self._get_goal_str(intent)

#		print("***** user status *****\n->", self.user_status, "\n")
#		print("***** current intent *****\n->", intent, "\n") # BACK
#		print("***** current intent constraint *****\n->", self.current_constraints[intent], "\n")
#		print("***** corresponding goal str *****\n->", goal_str, "\n")
		# print("***** current entities provided (empty) *****\n->", self.intent_entity_provided, "\n")
		return goal_str


	def _use_next_constraints(self, intent: str, slot_type: str) -> None:
		"""move the constraint pointer to the next"""
		# Another problem is that how to decide which slot type (search or book) to add when failure?
		# one solution is that dont use act mapping to keep NoOffer and NoBook separate, if so, try use nl on act
		self.user_status["constraint_idx"][intent][slot_type] += 1
		if self.user_status["constraint_idx"][intent][slot_type] >= self.n_max_value[slot_type]:
			# TODO: ask Alex, usually how to deal with this warning case? And make it as warning rather than just print
			print(f"Failure times on {slot_type} is more than the given value candidates, no new value to choose as alternative")
			print(f"A valid goal should not enter here!")
			self.user_status["constraint_idx"][intent][slot_type] = self.n_max_value[slot_type] - 1 # let user use last values as they are supposed to be fine


	def _update_current_constraints(self, intent: str, spk: str, usr_act: str, sys_act: str) -> None:
		# TODO: complete instruction here
		"""Update current constraints used for generation based on either previous usr or sys act

		:param act:
		:param spk:
		:param intent:
		:return:
		"""
		assert spk in ["usr", "sys"]
		# act_dict = parse_act(act)
		intent_constraints = self.current_constraints[intent]

		if spk == "sys":
			act_dict = self.parse_act(sys_act)
			##### when the system informs failure (search or book), use next set of constraints given in goal #####
			# if "_NOTIFY_FAILURE_" in act_dict:
			if self.const_act_str["fail_search"] in act_dict:
				slot_type = self.slot_types["search"]
				self._use_next_constraints(intent, slot_type)
				keep_slot_types = [self.slot_types["search"], self.slot_types["book"]] # still in search phase, book slots should be kept
				self._prepare_current_constraints([intent], keep_slot_types, if_reset_reqt=False) # only change constraints for this intent

			elif self.const_act_str["fail_book"] in act_dict:
				slot_type = self.slot_types["book"]
				self._use_next_constraints(intent, slot_type)
				keep_slot_types = [self.slot_types["book"]] # already found entities, no need to keep search slots
				self._prepare_current_constraints([intent], keep_slot_types, if_reset_reqt=False)

			##### when the system request #####
			elif self.const_act_str["request"] in act_dict:
				requested_slots = act_dict[self.const_act_str["request"]]
				for slot in requested_slots.keys():
					# requested slot in current constraint, do nothing
					if slot in intent_constraints["informable"].keys():
						continue

					# slots that are beyond the current goal enter the following section
					# case 1: requested slot in the complete goal,
					# this should be entered if the system requests the informed slots
					# if slot in self.complete_constraints[intent]["informable"].keys():
					# 	value = self.complete_constraints[intent]["informable"][slot]
					if slot in self.complete_goal["constraints"][intent]["informable"].keys(): # dict of slot to value_list
						slot_type = self._get_slot_type(slot)
						value_idx = self.user_status["constraint_idx"][intent][slot_type]
						value = self.complete_goal["constraints"][intent]["informable"][slot][value_idx]

					# case 2: requested slot not in the complete goal, set the value to "dontcare"
					# can sample a new value here for more interesting interactions
					else:
						value = "dontcare" # "no preference" # TODO: play around to see nlg output
					intent_constraints["informable"][slot] = value

		else: # usr
			act_dict = self.parse_act(usr_act)
			##### remove informed slot/value pair, if informed #####
			if self.const_act_str["inform"] in act_dict:
				for slot, value_list in act_dict[self.const_act_str["inform"]].items():
					# value = value_list[0]
					for value in value_list: # possible to have multi-value slots in user act in corpus
						if self.finish_inform == "loose" and slot in intent_constraints["informable"]:
							del intent_constraints["informable"][slot]
						if self.finish_inform == "strict" and slot in intent_constraints["informable"] and value == intent_constraints["informable"][slot]:
							del intent_constraints["informable"][slot]

			##### remove requested slot, if requested #####
			if self.const_act_str["request"] in act_dict:
				sys_act_dict = self.parse_act(sys_act) # auxiliary check
				for slot in act_dict[self.const_act_str["request"]].keys():
					# if slot in intent_constraints["requestable"]: # one choice
					if slot in intent_constraints["requestable"] and slot in sys_act_dict[self.const_act_str["inform"]].keys(): # another choice, more strict
						intent_constraints["requestable"].remove(slot)


	def _add_info(self, slot_to_value_list, slot, value) -> None:
		# print(slot)
		# assert slot in self.schema_slots # SLOT_FORMAT
		# constraints[intent]["informable"][slot] = value
		if slot not in slot_to_value_list:
			slot_to_value_list[slot] = []
		# assert value not in slot_to_value_list[slot]
		if value not in slot_to_value_list[slot]:
			slot_to_value_list[slot].append(value)
			slot_type = self._get_slot_type(slot)
			if len(slot_to_value_list) > self.n_max_value[slot_type]:
				self.n_max_value[slot_type] = len(slot_to_value_list)


	def _add_reqt(self, slot_set, slot) -> None:
		# assert slot in self.schema_slots # SLOT_FORMAT
		# constraints[intent]["requestable"].add(slot)
		slot_set.add(slot)


	def _validate_input_goal(self):
		"""validate the input goal"""
		# TODO: finish the method
		# assert all([intent in self.schema_intents for intent in intents]) # ensure intents are in schema
		pass


	@staticmethod
	def parse_act(act_seq: str) -> dict:
		"""parse usr/sys act string into dict(act: {slot=value_list}) (slots in act_request have '_Empty_' value) """
		act_dict = {}
		assert isinstance(act_seq, str)
		act_seq = act_seq.split("<ACT/>")
		for act_seg in act_seq:
			if act_seg == "":
				continue

			act_seg = act_seg.strip()  # remove space at the start/end
			act_seg = act_seg.split()
			##### get act in special token format #####
			# act = act_seg[0] # e.g., _INFORM_, _REQUEST_
			# assert act[0] == "_" and act[-1] == "_"
			# act_seg = " ".join(act_seg[2:]) # discard first two tokens, "_ACT_ </ACT>"

			##### get act in natural language format #####
			end_idx = act_seg.index("</ACT>")
			act = " ".join(act_seg[: end_idx])
			act_seg = " ".join(act_seg[end_idx + 1:])  # act arguments (slot/value pairs)
			# print(f"act: {act}\n", act_seg, "\n")

			assert act not in act_dict
			act_dict[act] = {}
			for sv_seg in act_seg.split("</VALUE>"):
				if sv_seg == "":
					continue

				sv_seg = sv_seg.replace("<SLOT/>", "")
				sv_seg = sv_seg.strip()  # remove spaces at begin and end
#				print("|{}|".format(sv_seg))
				slot, value = sv_seg.split("</SLOT> <VALUE/>")
				slot, value = slot.strip(), value.strip()
#				print("act: |{}|, slot: |{}|, value: |{}|".format(act, slot, value))
				# one slot one value
				# act_dict[act][slot] = value
				# one slot, multi-value is possible by system
				if slot not in act_dict[act]:
					act_dict[act][slot] = []
				if value not in act_dict[act][slot]:
					act_dict[act][slot].append(value)

		# print(act_dict)
		return act_dict


	def convert_into_system_act_format(self):
		# TODO
		pass
	##### below methods need be implemented for convlab-2 to work #####
	def init_session(self, **kwargs):
		"""Use this method to reset the agent state after each dialogue, if necessary.
        This gets called before each dialogue.

        Examples
        --------
        In `simulate_corpus_interaction.py` you will see that this is used, for example, to pass
        the dialogue to the corpus agent so it knows what to talk about.

        An example here would be to reset the dialogue context.
        """
		# dialogue goal in MultiWOZ2.1-like format
		self.current_goal = kwargs.get("ini_goal", {})
		self.policy.init_session(ini_goal=self.current_goal)
		self.current_goal = self.policy.get_goal()
		# TODO: ANYTHING ELSE THAT NEEDS TO HAPPEN BEFORE EACH DIALOGUE?
		self.context = []
		self.input_action = []
		self.output_action = []

		# init internal data
		self._context_str = "" # context string with special tags used in generation
		self._prev_usr_act = "" # user act string used in generation

		# goal process
		self.complete_goal = self._format_complete_goal(self.current_goal)
		self.user_status = self._init_user_status()
		self._get_scenario_str()
		self.current_constraints = {} # init
		self._prepare_current_constraints(self.complete_goal["intents"], list(self.slot_types.keys()), if_reset_reqt=True)
		print("input goal:\n", self.current_goal, "\n")
		print("complete goal:\n", self.complete_goal, "\n")
		print("current constraints:\n", self.current_constraints, "\n")
		# sys.exit(1)


	def response(self, sys_utterance: str) -> str:
		"""Generate natural language response given the system response.

		Parameters
		---------
		sys_utterance
			Last system utterance. For first turn, sys_utterance is the empty string.

		Returns
		-------
		response
			A natural language response.

		"""

		# TODO: MAKE APPROPRIATE USE OF THE HISTORY, BEHAVIOUR_PARAMS, CURRENT_GOAL, UPDATE_GOAL TO GENERATE A RESPONSE
		# TODO: DON'T FORGET TO UPDATE INPUT AND OUTPUT ACTIONS STATES
		# response = "I want Italian."
		gen_parse, gen_str = self.generate_whole_sequence(sys_utterance)
		self.update_internal_data(gen_parse) # prepare for next turn
		segment_gen(gen_str, "example dialogue") # crazyusermodel
		# TODO: update lists of context, da_in, da_out here
		return gen_parse["USR_UTT"]


	def get_in_da(self) -> List[List[str]]:
		"""Used by clients to retrieve the user model NLU.

		Returns
		-------
		NLU output, assumed to be a list of lists, each formatted as::

				[[intention, domain, slot, value], ...]

		Here ``intention`` refers to a dialogue act and the ``intention``, ``domain`` and ``slot`` strings should
		follow the same convention as the corpus dialogue act annotations (i.e., capitalised, and using the correct
		set of slot names).
		"""
		return self.input_action


	def get_out_da(self) -> List[List[str]]:
		"""Used by clients to retrieve the user model policy output.

		Returns
		-------
		Policy output, following the same convention as the NLU output.
		"""
		return self.output_action


	def get_reward(self) -> float:
		"""Dummy method, used for API consistency."""
		return -1


	def is_terminated(self) -> bool:
		"""This should tell an external client whether the user model considers they have completed the task."""
		# return False
		return self.user_status["dialogue_terminate"]


def parse_complete_gen(gen):
	"""parse the complete generation output, return predictions of system act, user act and user utterance"""
	output = {}
	for key in ["SYS_ACT", "SNT", "GC", "RA", "USR_ACT", "USR_UTT"]:
		value = find_segment(gen, key)
		output[key] = value
	# print("***** complete generation output *****\n->", gen, "\n") # BACK
	# print("***** parse output *****\n->", output, "\n")
	return output


def generate_example_goal() -> dict:
	"""create an example goal for testing"""
	# {service: service_meta},
	# service_mate: {"info": {slot: value}, "fail_info": {slot: value},
	# 				"book": {slot}: value, "fail_book": {slot: value}, "reqt": set(slot)}
	goal = {}
#	services = ["restaurant", "hotel"]
#	services = ["train", "attraction"]
	services = ["restaurant"]

#	# restaurant
	service = services[0]
	goal[service] = {}
	goal[service]["fail_info"] = {
		"food": "eastern european",
		"area": "south",
		"price range": "expensive",
	}
	goal[service]["info"] = {
		"food": "chinese",
		"area": "south",
		"price range": "cheap",
	}
	goal[service]["fail_book"] = {}
	goal[service]["book"] = {
		"book day": "monday",
		"book people": "8",
		"book time": "13:15"
	}
	goal[service]["reqt"] = {"address": "?"}

#	# hotel
#	service = services[1]
#	goal[service] = {}
#	goal[service]["fail_info"] = {
#		"stars": "3",
#		"price range": "cheap",
#		"area": "centre",
#		"internet": "_True_"
#	}
#	goal[service]["info"] = {
#		"stars": "5",
#		"price range": "expensive",
#		"area": "centre",
#		"internet": "_True_"
#	}
#	goal[service]["fail_book"] = {
#		"book day": "sunday",
#		"book stay": 3,
#		"book people": 2
#	}
#	goal[service]["book"] = {
#		"book day": "monday",
#		"book stay": 1,
#		"book people": 2
#	}
#	goal[service]["reqt"] = {"phone": "?", "postcode": "?"}

#	# train
#	service = services[1]
#	goal[service] = {}
#	goal[service]["info"] = {
#        "destination": "ely",
#        "day": "monday",
#        "arrive by": "19:00",
#        "departure": "cambridge",
#        "book people": "8"
#	}
#	goal[service]["reqt"] = {"duration": "?", "leave at": "?", "train id": "?"}

#	# attraction
#	service = services[1]
#	goal[service] = {}
#	goal[service]["info"] = {
#		"type": "college",
#		"area": "west"
#	}
#	goal[service]["reqt"] = {"phone": "?", "postcode": "?"}


	# taxi
#	service = services[0]
#	goal[service] = {}
#	goal[service]["info"] = {
#		"arrive by": "17:30",
#		"departure": "city stop restaurant",
#		"destination": "the cambridge punter"
#	}
#	goal[service]["reqt"] = {"phone": "?", "type": "?"}
	# more services...
	return goal


def interact(checkpoint_path):
	user_model = NeuralAgent("user", checkpoint_path, "config.yaml")

	for dial_id in range(3):
		print(f"In the dialogue {dial_id}")
		goal = generate_example_goal()
		user_model.init_session(ini_goal=goal)
		sys_utt = ""

		for turn_id in range(100):
			usr_utt = user_model.response(sys_utt)

			if user_model.is_terminated():
				print("Dialogue terminates!")
				break

			# next turn materials
			sys_utt = input("Enter system response here: ")


if __name__ == "__main__":
	if len(sys.argv) == 1:
		print("Wrong argument!")
		print("Usage: python multiwoz_interact.py checkpoint_path")
		sys.exit(1)

	checkpoint_path = sys.argv[1]
	interact(checkpoint_path)
