import sys, os
from tqdm import tqdm

from analysis_multiwoz import *
from utils_multiwoz import *
from utils_sgd import get_special_tokens, compare_slot_values_in_state, wrap_element, add_str, dict2list, conv_special_token

''' pre-process script for MultiWOZ v2.2 '''

class DialMetaData():
	def __init__(self, dial_id, dial_meta, dial_act, unify_act):
		self.dial_id = dial_id
		self.unify_act = unify_act
		self.turn_meta_list, self.scenario = self.parse(dial_meta, dial_act) # None for system turn
		self.linearise_turns()

	def parse(self, dial_meta, dial_act):
		global n, act_intent, non_intent 
		assert len(dial_meta["turns"]) == len(dial_act)

		turn_meta_list = []
		scenario = []
		sys_turn = None # dummy sys turn for first usr turn
		prev_intent = ""
		prev_usr_turn, prev_usr_turn_meta = None, None # dummpy for tracing goal change at first turn
		for turn_id, turn in enumerate(dial_meta["turns"]):
			assert turn_id == int(turn["turn_id"])

			if turn["speaker"] == "SYSTEM":
				sys_turn = turn
				turn_meta_list.append(None)
				continue

			# init turn meta
			turn_meta = TurnMetaData(prev_intent, sys_turn, turn, self.dial_id, self.unify_act)

			# get goal change label
			turn_meta.get_goal_change_label(prev_usr_turn, prev_usr_turn_meta)

			# update previous goal
			for prev_turn_meta in reversed(turn_meta_list):
				if prev_turn_meta is None:
					continue
				prev_turn_meta.accumulate_constraints(turn_meta) # TODO: check goal

			# record task (intent) in scenario
			if turn_meta.usr_intent not in scenario:
				scenario.append(turn_meta.usr_intent)

			turn_meta_list.append(turn_meta)
			prev_intent = turn_meta.usr_intent
			prev_usr_turn, prev_usr_turn_meta = turn, turn_meta
		assert len(turn_meta_list) == len(dial_meta["turns"])
		return turn_meta_list, scenario


	def linearise_turns(self):
		# linearise necessary meterials
		for turn_meta in self.turn_meta_list:
			if turn_meta is None:
				continue
			turn_meta._linearise(self.scenario, SERVICE2META)



class TurnMetaData():
	def __init__(self, prev_intent, sys_turn, usr_turn, dial_id, unify_act):
		self.dial_id = dial_id
		self.unify_act = unify_act
		self.original_act_set = get_original_act_set() # act set w/o domain information
		self.sys_turn, self.usr_turn = sys_turn, usr_turn

		# turn id
		self.sys_turn_id, self.usr_turn_id = self._get_turn_id(sys_turn, usr_turn)

		# intent
		self.usr_intent = normalise_intent(self._get_intent(usr_turn, prev_intent))
		if remove_book_intent:
			self.usr_intent = self.usr_intent.replace("book", "find")
		assert self.usr_intent in INTENTS # or self.usr_intent == "temp temp"
		self.service = self.usr_intent.split()[1]

		# utterances
		self.utt = {}
		self.utt["sys"], self.utt["usr"] = self._get_utt(sys_turn), self._get_utt(usr_turn)

		# act
		self.act2sv = {}
		self.act2sv["sys"], _ = self._parse_action(self.sys_turn_id, self.sys_turn)
		self.act2sv["usr"], self.usr_constraints = self._parse_action(self.usr_turn_id, self.usr_turn)

		# task boundary
		self._get_new_task_label(prev_intent)

		# req_alts
		self._get_req_alts_label()


	def _get_turn_id(self, sys_turn, usr_turn):
		usr_turn_id = int(usr_turn["turn_id"]) # 0, 2, 4 ...
		sys_turn_id = int(sys_turn["turn_id"]) if sys_turn is not None else -1
		assert sys_turn_id == (usr_turn_id-1)
		return sys_turn_id, usr_turn_id


	def _get_utt(self, turn):
		if turn == None:
			return ""
		return turn["utterance"]


	def accumulate_constraints(self, new_turn_meta):
		'''
			Add slot, slot-value pairs from a given following turn
			This function forms the user goal by accumulating constraints backward
		'''
		# only accumulate constraints with the same task/intent
		if new_turn_meta.usr_intent != self.usr_intent:
			return

		if new_turn_meta.goal_change: # if goal changes at a new turn, these constraints should not be put in previous turns
			return

		# only accumulate constraints without goal change
		# if the value of a slot is changed (goal change) in a new turn,
		# this slot-value pair is not part of initial goal and should not be added into the goal of previous turns
		new_constraints = new_turn_meta.usr_constraints
		self.usr_constraints["requestable"] = self.usr_constraints["requestable"].union(new_constraints["requestable"])
		for slot, value_list in new_constraints["informable"].items():
			if slot not in self.usr_constraints["informable"]:
				self.usr_constraints["informable"][slot] = value_list


	def get_goal_change_label(self, prev_usr_turn, prev_turn_meta):
		''' check if goal changed (value of slot changes) between two turn states '''
		# first usr turn
		if prev_usr_turn is None:
			assert self.usr_turn_id == 0
			self.goal_change = False
			return

		# last usr turn
		if "GOODBYE" in self.act2sv["usr"] or "THANK_YOU" in self.act2sv["usr"]:
			self.goal_change = False
			return

		assert self.usr_turn_id != 0
		assert prev_usr_turn["speaker"] == "USER"

		# new task
		if self.usr_intent != prev_turn_meta.usr_intent:
			self.goal_change = False
			return

		# compare two states to obtain goal change flag
		curr_state, prev_state = None, None
		for frame in self.usr_turn["frames"]:
			if frame["service"] == self.service:
				curr_state = frame["state"]["slot_values"]

		for frame in prev_usr_turn["frames"]:
			if frame["service"] == prev_turn_meta.service:
				prev_state = frame["state"]["slot_values"]

		# check if slot value has changed at current turn (new slot is not counted)
		assert curr_state is not None and prev_state is not None
		self.goal_change = compare_slot_values_in_state(curr_state, prev_state)


	def _get_domain_from_act(self, dialogue_act):
		'''
			parse the raw dialouge act annotation to get domain info
			number of doamin can be more than 1 for multi-domain turns
		'''
		domains = set()
		book_flag = False
		for dact, sv_pairs in dialogue_act.items():
			assert "-" in dact
			domain, _ = dact.split("-")
			if domain not in ["Booking", "general"]:
				domains.add(domain)
			for slot, value in sv_pairs:
				if "book" in slot: # e.g., bookday
					book_flag = True
		return domains, book_flag


	def _get_intent(self, usr_turn, prev_intent):
		intents = []
		for frame in usr_turn["frames"]:
			service = frame["service"]
			intent = frame["state"]["active_intent"]
			if intent != "NONE":
				intents.append(intent)

		if len(intents) == 1:
			intent = intents[0]
			if intent == "find_taxi": intent = "book_taxi"
			return intent # tackle 51.5k out of 71.5k user turns


		# if above fails (e.g., due to wrong label), leverage usr act to help determine main intent/service
		# possible domains in da: {'Hospital', 'Taxi', 'Train', 'Police', 'Restaurant', 'Booking', 'general', 'Attraction', 'Hotel'}
		usr_act = data_act[self.dial_id][str(self.usr_turn_id)]["dialog_act"]
		domains, book_flag = self._get_domain_from_act(usr_act)
		if len(domains) == 1:
			domain = list(domains)[0].lower()
			if book_flag and domain in ["restaurant", "hotel", "train"]:
				intent = "book_{}".format(domain)
			elif domain == "taxi":
				intent = "book_{}".format(domain)
			else:
				intent = "find_{}".format(domain)
			return intent # tackle 58.1k out of 71.5k user turns

		if "Taxi" in domains:
			return "book_taxi" # tackle 58.8k out of 71.5k user turns

		if self.usr_turn_id == 0: # wrong label at first turn, no previous intent to use, only 136 user turns here
			utt = usr_turn["utterance"]
			if "restaurant" in utt or "Restaurant" in utt or "eat" in utt or "din" in utt:
				return "find_restaurant"
			elif "hotel" in utt or "room" in utt or "house" in utt or "stay" in utt or "live" in utt:
				return "find_hotel"
			else:
				return "find_attraction" # tackle 58.9k out of 71.5k user turns

		else: # not first turn, leverage sys act to help decide intent
			sys_act = data_act[self.dial_id][str(self.sys_turn_id)]["dialog_act"]
			sys_domains, _ = self._get_domain_from_act(sys_act)
			if len(sys_domains) == 1:
				domain = list(sys_domains)[0].lower()
				if book_flag and domain in ["restaurant", "hotel", "train"]:
					intent = "book_{}".format(domain)
				elif domain == "taxi":
					intent = "book_{}".format(domain)
				else:
					intent = "find_{}".format(domain)
				return intent # tackle 67.3k out of 71.5k user turns
				
		# two cases left enter here
		# 1. turns with only general act, e.g., bye
		# 2. turns have multiple intents (very few)
		# both will be handled using previous intent
		assert prev_intent != ""
		intent = "_".join(prev_intent.split()) # as prev_intent has been normalised already
		return intent


	def _parse_action(self, turn_id, turn):
		'''parse the `dialog_act` field in `dialog_acts.json`

			Returns:
				act2sv: act to slot value pairs, {act=sv}; sv: slot to value list, {slot=[v1, v2]}
		'''
		act2sv = dict()
		constraints = {"informable": dict(), "requestable": set()}
		if turn is None:
			return None, constraints

		# get da from data_act
		dialogue_act = data_act[self.dial_id][str(turn_id)]["dialog_act"]
		domains = set()
		for dact, svs in dialogue_act.items():
			assert "-" in dact
			if self.unify_act: # will use only act part without domain info
				domain, act = dact.split("-") # split `domain-act`, e.g., `hotel-inform` -> hotel, inform
			else: # keep original mwoz act
				act = dact # use act with domain info

			if self.unify_act:
				# unify act: `Booking-Inform` with no args is equivalent to `OfferBook` in train domain
				if dact == "Booking-Inform" and svs == [['none', 'none']]:
					act = "OfferBook"

			# deal with act
			if self.unify_act:
				assert act in self.original_act_set
				if turn["speaker"] == "USER": assert act in ["Inform", "Request", "bye", "thank", "greet"]
				act = get_act_natural_language(act)

			if act not in act2sv:
				act2sv[act] = dict()
				
			# iterate slot value pairs
			for slot, value in svs:
				slot = normalise_slot(slot)
				value = normalise_value(value)

				# act to slot value pairs
				# NOTE: same slot might appear more than once per turn, e.g., when the system informs two hotels with their addresses
				# so a value list is stored for each slot
				if slot not in act2sv[act]:
					act2sv[act][slot] = []
				act2sv[act][slot].append(value)
	
				# collect constraints
				if act in ["REQUEST", "Request", "request"]:
					constraints["requestable"].add(slot)
				else:
					if slot != "Empty":
						if slot not in constraints["informable"]: # NOTE: same reason as act, value list per slot
							constraints["informable"][slot] = []
						constraints["informable"][slot].append(value)
		return act2sv, constraints


	def _linearise(self, scenario, service2meta):
		self.linear_act = {}
		self.linear_act["sys"] = self._linearise_act(self.act2sv["sys"])
		self.linear_act["usr"] = self._linearise_act(self.act2sv["usr"])
		self.linear_goal = self._linearise_goal(self.usr_constraints, scenario, service2meta)


	def _linearise_goal(self, constraints, scenario, service2meta):
		'''
			linearise goal representation which consists of several parts:
			scenario, task (intent), task description, constraints with informable and requestable
			e.g., <SCENARIO/> task1 task2 .. </SCENARIO>
				  <TASK/> current task </TASK> <DESC/> task description </DESC>
				  <INFORM/> <SLOT/> slot1 </SLOT> <VALUE> value1 </VALUE> .. </INFORM>
				  <REQUEST/> <SLOT> slot1 </SLOT> <SLOT> slot2 </SLOT> .. </REQUEST> 
		'''
		res = ""
		# scenario
		assert isinstance(scenario, list) and len(scenario) > 0
		scenario = " ".join([wrap_element("INTENT", intent) for intent in scenario]) # treat intent as nl
		scenario_wrap = wrap_element("SCENARIO", scenario)
		res = add_str(res, scenario_wrap)

		# task name
		intent = self.usr_intent
		assert intent in scenario
		intent_wrap = wrap_element("TASK", intent)
		res = add_str(res, intent_wrap)

		# task description
		description = service2meta[self.service]["intents"][intent]["description"]
		description_warp = wrap_element("DESC", description)
		res = add_str(res, description_warp)

		# informable
		informable = dict2list(constraints["informable"]) # sorted sv pair list [slot=value]
		res = add_str(res, "<INFORM/>")
		for sv_pair in informable:
			slot, value = sv_pair.split("=")
			if value in ["True", "False", "Empty"]:
				value = conv_special_token(value, SPECIAL_TOKENS)
			if slot in ["Empty"]:
				slot = conv_special_token(slot, SPECIAL_TOKENS)
			# slot
			slot_wrap = wrap_element("SLOT", slot)
			res = add_str(res, slot_wrap)
			# value
			value_wrap = wrap_element("VALUE", value)
			res = add_str(res, value_wrap)
		res = add_str(res, "</INFORM>")

		# requestable
		requestable = sorted(list(constraints["requestable"])) # sorted slot list [slot]
		res = add_str(res, "<REQUEST/>")
		for slot in requestable:
			slot_wrap = wrap_element("SLOT", slot)
			res = add_str(res, slot_wrap)
		res = add_str(res, "</REQUEST>")
		return res[1:] # remove first space


	def _linearise_act(self, act2sv):
		'''
		NOTE: 1) split slot/value if "_"; 2) special tokens of acts; 3) empty slot or empty value
		NOTE: filer too many values (e.g., 10 movie names) but make sure the one the user chose is present

		Return: ordered (slots sorted within act, acts sorted) linearised act sequence,
				e.g., <ACT/> <INFORM> </ACT> <SLOT/> area </SLOT> <VALUE/> Cambridge </VALUE> ...
				e.g., <ACT/> <REQUEST> </ACT> <SLOT/> _Empty_ </SLOT> <VALUE/> _Empty_ </VALUE>
		'''
		res = ""
		if act2sv is None:
			return res

		for act in sorted(act2sv.keys()): # sort act
			sv = act2sv[act] # dict{slot: value_list}
			act_wrap = wrap_element("ACT", act)
			res = add_str(res, act_wrap)

			sorted_sv = dict2list(sv) # sorted sv list, [s1=v1, s2=v2], note slot can repeat
			for sv_pair in sorted_sv:
				slot, value = sv_pair.split("=")
				if value in ["True", "False", "Empty"]:
					value = conv_special_token(value, SPECIAL_TOKENS)
				if slot in ["Empty"]:
					slot = conv_special_token(slot, SPECIAL_TOKENS)

				# slot
				slot_wrap = wrap_element("SLOT", slot)
				res = add_str(res, slot_wrap)

				# value
				value_wrap = wrap_element("VALUE", value)
				res = add_str(res, value_wrap)

		return res[1:] # remove first space


	def _get_new_task_label(self, prev_intent):
		'''
			get a binary label indicating if a turn starts a new task (intent) in dialogue
		'''
		assert prev_intent != "NONE" and self.usr_intent != "NONE"
		if self.usr_intent != prev_intent:
			self.start_new_task = True
		else:
			self.start_new_task = False

	def _get_req_alts_label(self):
		self.req_alts = False # no request alternative in mwoz


def collect_examples(dial_id, dial_meta, examples):
	num = 0
	examples[dial_id] = {}
	for turn_meta in dial_meta.turn_meta_list:
		if turn_meta is None: # sys turn
			continue

		example_id = "{}-{}".format(dial_id, num)
		example = {
			"utterances": turn_meta.utt,
			"actions": turn_meta.linear_act,
			"goal": turn_meta.linear_goal,
			"service": turn_meta.service,
			"intent": turn_meta.usr_intent,
			"goal_change": turn_meta.goal_change,
			"start_new_task": turn_meta.start_new_task,
			"req_alts": turn_meta.req_alts
		}
		examples[dial_id][example_id] = example
		num += 1


def prepare_data_seq(unify_act, out_data_path):
	for split in DATA_SPLIT:
		examples = {}
		for dial_num, dial_id in enumerate(tqdm(sorted(data[split].keys()))):
			dial = data[split][dial_id]
			dial_act = data_act[dial_id]

			dial_meta = DialMetaData(dial_id, dial, dial_act, unify_act)
			collect_examples(dial_id, dial_meta, examples)

		with open("{}/{}.json".format(out_data_path, split), "w") as f:
			json.dump(examples, f, sort_keys=True, indent=4)
		print("Done process {} {} dialogues".format(split, len(examples)))


if __name__ == "__main__":
	if len(sys.argv) == 1:
		print("Wrong argument!")
		print("usage: python utils/preprocess_multiwoz.py multiwoz2.2-data-path")
		sys.exit(1)

	# Set data path
	data_path = sys.argv[1]
	out_data_path = "./processed_data/multiwoz/"
	os.makedirs(out_data_path, exist_ok=True)

	# Control flags
	unify_act = True
	remove_book_intent = True

	# Load data and material as global var
	SERVICE2META, INTENTS, SLOTS = load_schema(os.path.join(data_path, "schema.json"))
	SPECIAL_TOKENS = get_special_tokens()
	data, data_act = collect_data(data_path, remove_dial_switch=False)

	prepare_data_seq(unify_act, out_data_path)
