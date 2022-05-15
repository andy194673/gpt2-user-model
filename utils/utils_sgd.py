import re
import json

''' This file contains utility functions for SGD '''

class bcolors:
	HEADER = '\033[95m'
	OKBLUE = '\033[94m'
	OKCYAN = '\033[96m'
	GREEN = '\033[92m'
	YELLOW = '\033[93m'
	RED = '\033[91m'
	ENDC = '\033[0m'
	BOLD = '\033[1m'
	UNDERLINE = '\033[4m'


def wrap_element(content_type, content):
	''' wrap elements such as slot, value, e.g., <SLOT/> slot </SLOT> '''
	assert "/" not in content_type
	return "<{}/> {} </{}>".format(content_type, content, content_type)


def add_str(str1, str2):
	return (str1 + " " + str2)


def list2str(x):
	x = sorted(x)
	x = ",".join(x)
	return "[" + x + "]"


def dict2str(x, colored=False):
	output = []
	for key, value in x.items():
		if isinstance(value, list):
			value = list2str(value)
		if colored:
			output.append(f"{bcolors.YELLOW}{key}{bcolors.ENDC}={bcolors.GREEN}{value}{bcolors.ENDC}")
		else:
			output.append("{}={}".format(key, value))
	return list2str(output)


def dict2list(x):
	output = []
	for key, value in x.items():
		assert isinstance(key, str)

		if isinstance(value, str):
			output.append("{}={}".format(key, value))

		elif isinstance(value, list): # only 1.8 turns, usually on system side, have multiple values for a slot
#			if len(value) > 2: print("************************HIT************************")
			for x in value:
				assert isinstance(x, str)
				output.append("{}={}".format(key, x))
	return sorted(output)
		

def compare_slot_values_in_state(slot_values1, slot_values2):
	''' return True if value in any intersection slot is different between two states '''
	for slot in (slot_values1.keys() & slot_values2.keys()): # check slots in intersection to see if their value changed
		values1 = slot_values1[slot]
		values2 = slot_values2[slot]
		if len(set(values1)&set(values2)) == 0: # none of values matched between two value lists
			return True
	return False


def get_turn_intent(turn):
	''' return turn's intent and service '''
	frames = turn["frames"]
	if len(frames) == 1:
		return frames[0]["state"]["active_intent"], frames[0]["service"]
	else:
		assert len(frames) == 2
		for frame in frames:
			for action in frame["actions"]:
				if action["act"] == "INFORM_INTENT":
					return frame["state"]["active_intent"], frame["service"]


def get_turn_act(turn):
	''' return an `act` list of a turn '''
	acts = []
	for frame in turn["frames"]:
		for actions in frame["actions"]:
			acts.append(actions["act"])
	return acts


def get_categorical_slot_span_info(slots):
	'''
		Inputs: list of dict in `slots` field annotation
	'''
	slot2info = {}
	for slot in slots:
		slot_name = slot["slot"]
		slot2info[slot_name] = slot
	return slot2info


def show_turn_meta(turn_meta):
	print(f"intent: {bcolors.RED}{turn_meta.usr_intent}{bcolors.ENDC}, start_new_task: {bcolors.OKBLUE}{turn_meta.start_new_task}{bcolors.ENDC}, goal_change: {bcolors.OKCYAN}{turn_meta.goal_change}{bcolors.ENDC}")
	show_constraints(turn_meta.usr_constraints)
	print("#####"*10)
	show_linear_data(turn_meta.linear_goal, tag="Goal")
	show_linear_data(turn_meta.linear_act["sys"], tag="SYS act")
	show_linear_data(turn_meta.linear_act["usr"], tag="USR act")
	print("#####"*10)
	print("")


def show_constraints(usr_constraints):
	info, req = usr_constraints["informable"], usr_constraints["requestable"]
	info = dict2str(info, colored=True)
	print("informable: {}, requestable: ".format(info), end="")
	for x in req:
		print(f"{bcolors.YELLOW}{x}{bcolors.ENDC}", end=" ")
	print("")

	
def show_linear_data(data, tag):
	print("{}: |{}|".format(tag, data))


def load_schema(data_path):
	''' load schema and return (1) dict {service: service content}, (2) set of intents, and (3) set of slots '''
	def _update(key, value, mapping):
		if key in mapping:
			assert value == mapping[key] # ensure service meta is the same between data splits
		else:
			mapping[key] = value

	def _restructure_service_meta(service_meta, attribute):
		''' convert slot/intent list into dict(name=meta) '''
		assert attribute in ["slots", "intents"]
		mapping = {}
		for value in service_meta[attribute]:
			key = value["name"]
			mapping[key] = value
		service_meta[attribute] = mapping

	SERVICE2META = {}
	SLOTS, INTENTS = set(), set()
	for split in ["train", "dev", "test"]:
		with open("{}/{}/schema.json".format(data_path, split)) as f:
			data = json.load(f)

		for service_meta in data:
			service = service_meta["service_name"]
			_restructure_service_meta(service_meta, "slots")
			_restructure_service_meta(service_meta, "intents")
			_update(service, service_meta, SERVICE2META)

			for slot in service_meta["slots"]:
				SLOTS.add(slot)
			for intent in service_meta["intents"]:
				INTENTS.add(intent)
			# NOTE: the slot/intent existing in different services have different meta data, e.g., FindBus, event_name
	print("Load schema, intents: {}, slots: {}".format(len(INTENTS), len(SLOTS)))
	return SERVICE2META, INTENTS, SLOTS


def get_special_tokens():
	''' get pre-defined special tokens '''
	SPECIAL_TOKENS = {
		"bos_token": "<BOS>",
		"eos_token": "<EOS>",
		"pad_token": "<PAD>",
		"sep_token": "<SEP>",
		"additional_special_tokens": []
	}

	# ctx
	SPECIAL_TOKENS["additional_special_tokens"] += ["<CTX/>", "<USR/>", "<SYS/>"]
	SPECIAL_TOKENS["additional_special_tokens"] += ["</CTX>", "</USR>", "</SYS>"]

	# current turn utterance
	SPECIAL_TOKENS["additional_special_tokens"] += ["<USR_UTT/>", "<SYS_UTT/>"]
	SPECIAL_TOKENS["additional_special_tokens"] += ["</USR_UTT>", "</SYS_UTT>"]

	# current turn action
	SPECIAL_TOKENS["additional_special_tokens"] += ["<USR_ACT/>", "<SYS_ACT/>"]
	SPECIAL_TOKENS["additional_special_tokens"] += ["</USR_ACT>", "</SYS_ACT>"]

	# elements segment
	SPECIAL_TOKENS["additional_special_tokens"] += ["<ACT/>", "<SLOT/>", "<VALUE/>"]
	SPECIAL_TOKENS["additional_special_tokens"] += ["</ACT>", "</SLOT>", "</VALUE>"]


	# goal (`task` is `intent` in SGD)
	SPECIAL_TOKENS["additional_special_tokens"] += ["<GOAL/>", "<SCENARIO/>", "<TASK/>", "<DESC/>", "<INFORM/>", "<REQUEST/>"]
	SPECIAL_TOKENS["additional_special_tokens"] += ["</GOAL>", "</SCENARIO>", "</TASK>", "</DESC>", "</INFORM>", "</REQUEST>"]

	# sgd act
	SPECIAL_TOKENS["additional_special_tokens"] += ["_INFORM_", "_REQUEST_", "_CONFIRM_", "_OFFER_", "_NOTIFY_SUCCESS_", \
				"_NOTIFY_FAILURE_", "_INFORM_COUNT_", "_OFFER_INTENT_", "_REQ_MORE_", "_GOODBYE_", "_INFORM_INTENT_", \
				"_NEGATE_INTENT_", "_AFFIRM_INTENT_", "_AFFIRM_", "_NEGATE_", "_SELECT_", "_REQUEST_ALTS_", "_THANK_YOU_"]

	# multiwoz act, distinct from sgd
#	SPECIAL_TOKENS["additional_special_tokens"] += ["_RECOMMEND_", "_OFFER_BOOK_", "_GREET_", "_WELCOME_"]
	SPECIAL_TOKENS["additional_special_tokens"] += ["_OFFER_BOOK_", "_GREET_", "_WELCOME_"]

	# intent
	SPECIAL_TOKENS["additional_special_tokens"] += ["<INTENT/>"]
	SPECIAL_TOKENS["additional_special_tokens"] += ["</INTENT>"]

	# special values: True, False, Empty
	SPECIAL_TOKENS["additional_special_tokens"] += ["_True_", "_False_", "_Empty_"]
	# NOTE: slots, values and a special value "dontcare" are all presented in natural language

	# special flags
	# SNT: Start New Task
	# GC: Goal Change
	# RA: Request Alternative
	SPECIAL_TOKENS["additional_special_tokens"] += ["<SNT/>", "<GC/>", "<RA/>"]
	SPECIAL_TOKENS["additional_special_tokens"] += ["</SNT>", "</GC>", "</RA>"]

	print("Load special tokens: {}".format(len(SPECIAL_TOKENS["additional_special_tokens"])+4))
	return SPECIAL_TOKENS


def split_intent(intent):
	'''
		convert intent special token into a natural language (nl) form
		e.g., `FindEvents` -> `find events`
	'''
	assert intent[0].isupper()
	tokens = re.findall('[A-Z][^A-Z]*', intent) # e.g., `FindEvents` -> `Find Events`
	tokens = list(map(str.lower, tokens)) # lower case, -> `find events`
	intent_nl = " ".join(tokens)
	return intent_nl


def conv_special_token(token, SPECIAL_TOKENS):
	assert token[0] != "_" and token[-1] != "_"
	token = "_{}_".format(token)
	assert token in SPECIAL_TOKENS["additional_special_tokens"]
	return token
