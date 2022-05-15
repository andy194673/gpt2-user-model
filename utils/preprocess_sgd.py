import os, sys
from tqdm import tqdm
from analysis_sgd import *
from utils_sgd import *

'''pre-processing script for SGD

The annotations for a turn are grouped into frames, where each frame corresponds to a single service
The values of "slot_values" in user "state" is a list, where spoken variations are considered, e.g., tomorrow, 8/2
'''

class DialMetaData():
	def __init__(self, dial_id, dial):
		self.dial_id = dial_id
		self.turn_meta_list, self.scenario = self.parse(dial) # None for system turn
		self.linearise_turns()


	def parse(self, dial):
		turn_meta_list = []
		scenario = []
		sys_turn = None # dummy sys turn for first usr turn 
		prev_intent = ""
		prev_usr_turn, prev_usr_turn_meta = None, None # dummpy for tracing goal change at first turn
		for turn_id, turn in enumerate(dial["turns"]):
			if turn["speaker"] == "SYSTEM":
				sys_turn = turn
				turn_meta_list.append(None)
				continue

			# init turn meta
			turn_meta = TurnMetaData(prev_intent, sys_turn, turn, self.dial_id)

			# get goal change label
			turn_meta.get_goal_change_label(prev_usr_turn, prev_usr_turn_meta)

			# update previous goal
			for prev_turn_meta in reversed(turn_meta_list):
				if prev_turn_meta is None:
					continue
				prev_turn_meta.accumulate_constraints(turn_meta)

			# record task (intent) in scenario
			prev_intent = turn_meta.usr_intent
			if turn_meta.usr_intent not in scenario:
				scenario.append(turn_meta.usr_intent)

			turn_meta_list.append(turn_meta)
			prev_usr_turn, prev_usr_turn_meta = turn, turn_meta

		assert len(turn_meta_list) == len(dial["turns"])
		return turn_meta_list, scenario


	def linearise_turns(self):
		# linearise necessary meterials
		for turn_meta in self.turn_meta_list:
			if turn_meta is None:
				continue
			turn_meta._linearise(self.scenario)


class TurnMetaData():
	def __init__(self, prev_intent, sys_turn, usr_turn, dial_id):
		self.dial_id = dial_id
		self.sys_turn, self.usr_turn = sys_turn, usr_turn
		self.empty_token = "_Empty_"
		assert self.empty_token in SPECIAL_TOKENS["additional_special_tokens"]

		# intent
		self.usr_intent, self.service = self._get_intent(usr_turn, prev_intent)

		# utterances
		self.utt = {}
		self.utt["sys"], self.utt["usr"] = self._get_utt(sys_turn), self._get_utt(usr_turn)

		# action
		self.act2sv = {}
		self.act2sv["sys"], _ = self._parse_action(sys_turn)
		self.act2sv["usr"], self.usr_constraints = self._parse_action(usr_turn)

		# task boundary
		self._get_new_task_label(prev_intent)

		# req_alts
		self._get_req_alts_label(self.act2sv["usr"])


	def _get_intent(self, turn, prev_intent):
		''' manually set the `NONE` intent to the intent of previous turn '''
		active_intent, service = get_turn_intent(turn) # intent annotation (migt be `NONE`)
		if active_intent == "NONE":
			active_intent = prev_intent
		return active_intent, service


	def _get_utt(self, turn):
		if turn == None:
			return ""
		return turn["utterance"]


	def _parse_action(self, turn):
		'''
			parse action annotation to collect turn level information
			1) act to slot-value pairs, dict{act: {slot: value}}
			2) turn level constraints, dict{'informable': dict{slot: value}, 'requestable': set(slot)}
		'''
		# get mapping from act to slot-value pairs
		act2sv = {}
		info_req = {"informable": dict(), "requestable": set()} # constraints

		if turn == None:
			return None, info_req

		for frame in turn["frames"]:
			for action in frame["actions"]:
				act, slot, values = action["act"], action["slot"], action["values"]

				# deal with empty slot or value
				if turn["speaker"] == "USER": assert len(values) in [0, 1]
				if slot == "": slot = self.empty_token
				value = values[0] if len(values) > 0 else self.empty_token

				# act to slot-value pairs
				if act not in act2sv:
					act2sv[act] = {}
				assert slot not in act2sv[act]
				act2sv[act][slot] = value

				# collect constraints
				if slot in ["", self.empty_token]: # only act but no constraints, e.g., AFFIRM, NEGATE
					continue

				# turn level informalable and requestable info
				if act == "REQUEST":
					assert slot != ""
					info_req["requestable"].add(slot)
				else:
					if turn["speaker"] == "USER": assert act in ["INFORM_INTENT", "INFORM", "SELECT"] # not apply to system side
					if act != "SELECT": # result offered by system is part of initial user goal
						assert slot not in info_req["informable"]
						info_req["informable"][slot] = value
		return act2sv, info_req


	def accumulate_constraints(self, new_turn_meta):
		'''
			Add slot, slot-value pairs from a given following turn
			This function is used to form user goal by accumulating constraints backward
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
		for slot, value in new_constraints["informable"].items():
			if slot not in self.usr_constraints["informable"]:
				self.usr_constraints["informable"][slot] = value


	def _get_new_task_label(self, prev_intent):
		''' get a binary label indicating if a turn starts a new task (intent) in dialogue '''
		assert prev_intent != "NONE" and self.usr_intent != "NONE"
		if self.usr_intent != prev_intent:
			self.start_new_task = True
		else:
			self.start_new_task = False


	def _get_req_alts_label(self, act2sv):
		''' get a binary label indicating if usr requests alternatives '''
		if "REQUEST_ALTS" in act2sv:
			self.req_alts = True
		else:
			self.req_alts = False


	def get_goal_change_label(self, prev_usr_turn, prev_turn_meta):
		''' check if goal changed (value of slot changes) between two turn states '''
		if prev_usr_turn is None: # first usr turn
			self.goal_change = False
			return

		if len(self.usr_turn["frames"]) == 1 and self.usr_turn["frames"][0]["state"]["active_intent"] == "NONE": # `NONE` intent
			self.goal_change = False
			return

		if self.usr_intent != prev_turn_meta.usr_intent: # new task
			self.goal_change = False
			return

		assert prev_usr_turn["speaker"] == "USER"
		prev_state_sv, curr_state_sv = None, None
		for frame in prev_usr_turn["frames"]:
			if frame["state"]["active_intent"] == self.usr_intent:
				prev_state_sv = frame["state"]["slot_values"]

		# fix some weird cases (count very few, around 30 turns)
		if prev_state_sv is None:
			assert len(prev_usr_turn["frames"]) == 1 and prev_usr_turn["frames"][0]["state"]["active_intent"] == "NONE"
			prev_state_sv = prev_usr_turn["frames"][0]["state"]["slot_values"]

		for frame in self.usr_turn["frames"]:
			if frame["state"]["active_intent"] == self.usr_intent:
				curr_state_sv = frame["state"]["slot_values"]

		assert prev_state_sv is not None and curr_state_sv is not None
		self.goal_change = compare_slot_values_in_state(prev_state_sv, curr_state_sv) # True if goal changes


	def _linearise(self, scenario):
		self.linear_act = {}
		self.linear_act["sys"] = self._linearise_act(self.act2sv["sys"])
		self.linear_act["usr"] = self._linearise_act(self.act2sv["usr"])
		self.linear_goal = self._linearise_goal(self.usr_constraints, scenario)


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
			sv = act2sv[act] # dict{slot: value}

			act = "_{}_".format(act) # act is special token
			assert act in SPECIAL_TOKENS["additional_special_tokens"]
			act_wrap = wrap_element("ACT", act)
			res = add_str(res, act_wrap)

			sorted_sv = dict2list(sv) # sorted sv list, [slot=value]
			for sv_pair in sorted_sv:
				slot, value = sv_pair.split("=")
				slot, value = self._basic_normalise_slot(slot), self._basic_normalise_value(value, slot)

				# slot
				slot_wrap = wrap_element("SLOT", slot)
				res = add_str(res, slot_wrap)

				# value
				value_wrap = wrap_element("VALUE", value)
				res = add_str(res, value_wrap)
		return res[1:] # remove first space


	def _basic_normalise_value(self, value, slot):
		# intent value
		if slot == "intent":
			value = split_intent(value)
			return value

		# special token value
		if value in ["True", "False"]: # Empty is already in the form of "_Empty_"
			value = "_{}_".format(value)
			assert value in SPECIAL_TOKENS["additional_special_tokens"]
			return value
		return value


	def _basic_normalise_slot(self, slot):
		if slot not in SPECIAL_TOKENS["additional_special_tokens"]:
			slot = slot.replace("_", " ") # e.g., `date_of_journey` -> `date of journey`
		return slot


	def _linearise_goal(self, constraints, scenario):
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
		scenario = " ".join([wrap_element("INTENT", split_intent(intent)) for intent in scenario])
		scenario_wrap = wrap_element("SCENARIO", scenario)
		res = add_str(res, scenario_wrap)

		# task name
		intent = split_intent(self.usr_intent)
		assert intent in scenario
		intent_wrap = wrap_element("TASK", intent)
		res = add_str(res, intent_wrap)

		# task description
		description = SERVICE2META[self.service]["intents"][self.usr_intent]["description"]
		description_warp = wrap_element("DESC", description)
		res = add_str(res, description_warp)

		# informable
		informable = dict2list(constraints["informable"]) # sorted sv pair list [slot=value]
		res = add_str(res, "<INFORM/>")
		for sv_pair in informable:
			slot, value = sv_pair.split("=")
			slot, value = self._basic_normalise_slot(slot), self._basic_normalise_value(value, slot)
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
			slot = self._basic_normalise_slot(slot)
			slot_wrap = wrap_element("SLOT", slot)
			res = add_str(res, slot_wrap)
		res = add_str(res, "</REQUEST>")
		return res[1:] # remove first space


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


def prepare_data_seq(data, out_data_path):
	for split in DATA_SPLIT:
		examples = {}
		for dial_num, dial_id in enumerate(tqdm(sorted(data[split].keys()))):
			dial = data[split][dial_id]
			dial_meta = DialMetaData(dial_id, dial)
			collect_examples(dial_id, dial_meta, examples)

		with open("{}/{}.json".format(out_data_path, split), "w") as f:
			json.dump(examples, f, sort_keys=True, indent=4)
		print("Done process {} {} dialogues".format(split, len(examples)))


if __name__ == "__main__":
	if len(sys.argv) == 1:
		print("wrong arguments!")
		print("usage: python utils/preprocess_sgd.py sgd-data-path")
		sys.exit(1)

	# Set data path
	data_path = sys.argv[1]
	out_data_path = "./processed_data/sgd/"
	os.makedirs(out_data_path, exist_ok=True)

	# Load data and material as global var
	SERVICE2META, INTENTS, SLOTS = load_schema(data_path)
	SPECIAL_TOKENS = get_special_tokens()
	data = collect_data(data_path, remove_dial_switch=True)

	# Process data
	prepare_data_seq(data, out_data_path)
