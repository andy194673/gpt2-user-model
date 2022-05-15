import os
import json
from utils_sgd import list2str, dict2str, bcolors, compare_slot_values_in_state

''' This file contains some utilities for analysis and parsing SGD '''

DATA_SPLIT = ['train', 'dev', 'test']


def collect_data(data_path, remove_dial_switch=False):
	data = {}
	for split in DATA_SPLIT:
		data[split] = iter_data_folder(data_path, split, remove_dial_switch)
	return data


def _remove_dial(dial_id, dial):
	remove_flag = False
	# removes service `Homes_2` in test set as the slot `intent` is the same name as the user intent, which causes problem in goal preparation
	if "Homes_2" in dial["services"]:
		return True
	return False


def iter_data_folder(data_path, split, remove_dial_switch):
	''' Iterate data split folder '''
	split_dir = os.path.join(data_path, split)
	data_split = {}
	remove_dial_ids = []
	total_dial_ids = []
	for f in os.listdir(split_dir):
		if not f.startswith('dialogues'): # skip schema.json
			continue
		file_path = os.path.join(data_path, split, f)
		iter_file(file_path, data_split, remove_dial_ids, total_dial_ids, remove_dial_switch)
	print('Done collecting {} | total {} dialogues | load {} dialogues | remove {} dialogues'.format(split, len(total_dial_ids), len(data_split), len(remove_dial_ids)))
	return data_split


def iter_file(file_path, data_split, remove_dial_ids, total_dial_ids, remove_dial_switch):
	''' Iterate data file '''
	with open(file_path) as f:
		data_in = json.load(f) # list of dialouges in a json file

	for dial in data_in:
		dial_id = dial['dialogue_id']
		total_dial_ids.append(dial_id)

		if remove_dial_switch and _remove_dial(dial_id, dial):
			remove_dial_ids.append(dial_id)
		else:
			data_split[dial_id] = dial


def check_multiple_services_per_turn(data):
	for split in DATA_SPLIT:
		for dial_id in sorted(data[split].keys()):
			dial = data[split][dial_id]
			for turn_id, turn in enumerate(dial['turns']):
				frames = turn['frames']
				if len(frames) > 1:
					print(split, dial_id, turn_id, turn['utterance'])


def show_actions(actions):
	for action_id, action in enumerate(actions):
		act, slot, values = action["act"], action["slot"], action["values"]
		print(f"====> ACTION | Act {action_id}: {bcolors.RED}{act}{bcolors.ENDC}, slot: {bcolors.YELLOW}{slot}{bcolors.ENDC}, values: {bcolors.GREEN}{values}{bcolors.ENDC}")


def show_user_state(frame):
	state = frame["state"]
	active_intent = state["active_intent"]
	req_slots = list2str(state["requested_slots"])
	slot2value = dict2str(state["slot_values"], colored=True)
	print("====> STATE | intent: {}, req_slots: {}, slot2value: {}".format(active_intent, req_slots, slot2value))


def show_service_call(frame):
	if "service_call" not in frame:
		return
	# system calls api
	service_call, service_results = frame["service_call"], frame["service_results"]
	print("====> API call | method: {}, args: {}, results: {}".format(service_call["method"], dict2str(service_call["parameters"]), len(service_results)))


def show_frame(spk, frame_id, frame):
	service = frame["service"]
	print("==> Frame_id: {}, service: {}".format(frame_id, service))

	# actions (include all slots)
	show_actions(frame["actions"])

	# slots (only provide non-categorical slots with word span boundaries)
	if spk == "USER":
		show_user_state(frame)
	else: # system
		show_service_call(frame)


def show_turn(turn_id, turn):
	if turn is None:
		return

	frames = turn['frames']
	spk = turn['speaker']
	utt = turn['utterance']
	assert spk in ["USER", "SYSTEM"]
	print(f"{spk}: {bcolors.UNDERLINE}{utt}{bcolors.ENDC}")
	for frame_id, frame in enumerate(frames):
		show_frame(spk, frame_id, frame)
	print("------"*15)


def show_dial_info(dial_id, dial):
	print("\n") 
	print("******"*15)
	print("Dialogue={} | Service={}".format(dial_id, list2str(dial["services"])))
	print("******"*15)


def show_dial(dial_id, dial):
	show_dial_info(dial_id, dial)
	for turn_id, turn in enumerate(dial['turns']):
		show_turn(turn_id, turn)


def show_data(data):
	for split in DATA_SPLIT:
		for dial_id in sorted(data[split].keys()):
			dial = data[split][dial_id]
			show_dial(dial_id, dial)
			input("press...")


def identify_scenarios(data):
	'''
		According to dataset paper, a scenario is a sequence of intents, seeded at the start of a conversation
		to the user agent
	'''
	# TODO: deal with NONE intent, check the # of intent seq conbinations
	for split in DATA_SPLIT:
		scenario2dialogues = {}
		n_scenario_max, n_scenario_min = 0, 100
		for dial_id in sorted(data[split].keys()):
			dial = data[split][dial_id]
			scenario = []
			for turn in dial["turns"]:
				if turn["speaker"] == "SYSTEM":
					continue
				# USER turn
				# it's fine to consider only first frame (service) if the turn is at the bounrary between two services
				frame = turn["frames"][0]
				intent = frame["state"]["active_intent"]
				if intent == "NONE":
					continue
				if len(scenario) == 0 or intent != scenario[-1]:
					scenario.append(intent)

			# update count
			if len(scenario) > n_scenario_max: n_scenario_max = len(scenario)
			if len(scenario) < n_scenario_min: n_scenario_min = len(scenario)

			scenario = list2str(scenario)
			if scenario not in scenario2dialogues:
				scenario2dialogues[scenario] = []
			scenario2dialogues[scenario].append(dial_id)

		# done iter over split
		print("Summary: split={}, unique_scenario={}, max_intent={}, min_intent={}".format(split, len(scenario2dialogues), n_scenario_max, n_scenario_min))


def _check_request_alts_type(prev_turn, sys_turn, curr_turn, curr_acts):
	'''
		check which of the following happens when request_alts
		1. randomly change goal (state changes)
		2. request_alts as system provides venue with missing slot-value (usr provides new info)
		3. simply dislike the provided venue, change venue without new slot-value (same info)

		Input:
			prev_turn: previous user turn
			curr_turn: current user turn
	'''
	def _get_intent2state(turn):
		intent2state = {}
		for frame in turn["frames"]:
			state = frame["state"]
			intent = state["active_intent"]
			intent2state[intent] = state
		return intent2state
		
	assert "REQUEST_ALTS" in curr_acts
	if len(curr_acts) == 1: # case 3
#		return "_dislike_"
		if "OFFER" in get_turn_act(sys_turn):
			return "_dislike_offer_"
		else:
			return "_dislike_info_"
	elif "INFORM" in curr_acts and len(set(curr_acts)) == 2: # only inform and request_alts
		assert len(curr_turn["frames"]) == 1
		curr_slot_values = curr_turn["frames"][0]["state"]["slot_values"]
		curr_intent = curr_turn["frames"][0]["state"]["active_intent"]
		
		if len(prev_turn["frames"]) == 1:
			prev_slot_values = prev_turn["frames"][0]["state"]["slot_values"]
		else: # need to get the state with the same intent
			intent2state =  _get_intent2state(prev_turn)
			prev_slot_values = intent2state[curr_intent]["slot_values"]

		state_diff =  compare_slot_values_in_state(prev_slot_values, curr_slot_values)
		if state_diff: # case 1
			return "_random_"
		else: # case 2
			return "_miss_"
	else:
		return "_unknown_"


def stats_request_alts_type():
	for split in DATA_SPLIT:
		stats = {"_random_": 0, "_miss_": 0, "_dislike_offer_": 0, "_dislike_info_": 0, "_unknown_": 0}
		n_all_usr_turn, n_request_alts = 0, 0

		for dial_id in sorted(data[split].keys()):
			dial = data[split][dial_id]
			for turn_id, turn in enumerate(dial["turns"]):
				if turn["speaker"] == "SYSTEM":
					sys_turn = turn
					continue
				acts = get_turn_act(turn)
				if "REQUEST_ALTS" in acts:
					n_request_alts += 1
					type_result = _check_request_alts_type(prev_turn, sys_turn, turn, acts)
					stats[type_result] += 1
					if type_result == "_random_":
						print("CASE {}".format(type_result))
						show_turn(0, prev_turn)
						show_turn(0, sys_turn)
						show_turn(0, turn)
						input("press...")
				n_all_usr_turn += 1
				prev_turn = turn

		print("REQUEST_ALTS type statistics")
		for k, v in stats.items():
			print("{} => {}".format(k, v))
		print("request_alts turns: {}, all usr turns: {}, dialogues: {}".format(n_request_alts, n_all_usr_turn, len(data[split])))


def show_utt_by_act():
	target_act = "OFFER"
	for split in DATA_SPLIT:
		for dial_id in sorted(data[split].keys()):
			dial = data[split][dial_id]
			match_flag = False
			for turn_id, turn in enumerate(dial["turns"]):
				acts = get_turn_act(turn)
				if target_act in acts:
					match_flag = True
			if match_flag:
				show_dial(dial_id, dial)
				input('press...')


def show_state_with_value_change():
	for split in DATA_SPLIT:
		for dial_id in sorted(data[split].keys()):
			dial = data[split][dial_id]
			intent2slot_values = {}
			for turn_id, turn in enumerate(dial["turns"]):
				utt, spk = turn["utterance"], turn["speaker"]
				if spk != "USER":
					prev_system_turn = turn
					continue
				for frame in turn["frames"]:
					state = frame["state"]
					active_intent = state["active_intent"]
					slot_values = state["slot_values"]
					if active_intent in intent2slot_values:
						state_diff = compare_slot_values_in_state(intent2slot_values[active_intent], slot_values)
						if state_diff:
							print("Dial: {}, state change: {}".format(dial_id, state_diff))
							print("==> Prev SYS: {}".format(prev_system_turn["utterance"]))
							for sys_frame in prev_system_turn["frames"]:
								show_actions(sys_frame["actions"])
							print("==> Curr USR: {}".format(utt))
							show_actions(frame["actions"])
							print("recorded state => intent: {}, slot2value: {}".format(active_intent, dict2str(intent2slot_values[active_intent])))
							print("current  state => intent: {}, slot2value: {}".format(active_intent, dict2str(slot_values)))
							input("press...")
					intent2slot_values[active_intent] = slot_values # overlap with new state, no matter values changed or not


def check_state_with_value_change(display=False):
	for split in DATA_SPLIT:
		n_diff = {"NOTIFY_FAILURE": 0, "NEGATE": 0, "REQUEST_ALTS": 0, "RANDOM": 0}
		for dial_id in sorted(data[split].keys()):
			dial = data[split][dial_id]
			intent2slot_values = {}
			diff_flag = False
			for turn_id, turn in enumerate(dial["turns"]):
				if diff_flag:
					break
				utt, spk = turn["utterance"], turn["speaker"]
				if spk != "USER":
					prev_system_turn = turn
					continue
				for frame in turn["frames"]:
					state = frame["state"]
					active_intent = state["active_intent"]
					slot_values = state["slot_values"]
					if active_intent in intent2slot_values:
						state_diff = compare_slot_values_in_state(intent2slot_values[active_intent], slot_values)
						if state_diff:
							usr_acts = get_turn_act(turn)
							if "NOTIFY_FAILURE" in get_turn_act(prev_system_turn):
								if display: print('FAILURE', dial_id, utt)
								n_diff["NOTIFY_FAILURE"] += 1
							elif "NEGATE" in usr_acts:
								if display: print('NEGATE', dial_id, utt)
								n_diff["NEGATE"] += 1
							elif "REQUEST_ALTS" in usr_acts:
								if display: print('REQUEST_ALTS', dial_id, utt)
								n_diff["REQUEST_ALTS"] += 1
							else:
								if display: print('RANDOM', dial_id, utt)
								n_diff["RANDOM"] += 1
							if display: input("press...")
#							n_diff += 1
							diff_flag = True
					intent2slot_values[active_intent] = slot_values # overlap with new state, no matter values changed or not
		n = n_diff["NOTIFY_FAILURE"] + n_diff["NEGATE"] + n_diff["REQUEST_ALTS"] + n_diff["RANDOM"]
		print("{} => total dials: {}, change goal dials: {} (total: {})".format(split, len(data[split]), dict2str(n_diff), n))


def stats_after_system():
	'''
		check the possible user behavior right after system offers/notify_failure
	'''
	n = 0
	stats = {"SELECT": 0, "REQUEST_ALTS": 0, "REQUEST": 0, "AFFIRM": 0, "unknown": 0} # if system offers
#	stats = {"INFORM": 0, "AFFIRM": 0, "NEGATE": 0, "unknown": 0} # if system notify_failure
	for split in DATA_SPLIT:
		for dial_id in sorted(data[split].keys()):
			dial = data[split][dial_id]
			for turn_id, turn in enumerate(dial["turns"]):
				if turn_id == 0:
					prev_turn = turn
					continue
				if turn["speaker"] == "SYSTEM":
					sys_turn = turn
					continue

				if "OFFER" in get_turn_act(sys_turn):
#				if "OFFER" in get_turn_act(sys_turn) and "NOTIFY_FAILURE" in get_turn_act(sys_turn): 
#				if "NOTIFY_FAILURE" in get_turn_act(sys_turn):
					n += 1
					acts = get_turn_act(turn)
					# OFFER
					if "SELECT" in acts:
						stats["SELECT"] += 1
					elif "REQUEST_ALTS" in acts:
						stats["REQUEST_ALTS"] += 1
					elif "REQUEST" in acts:
						stats["REQUEST"] += 1
					elif "AFFIRM" in acts: # cases fall into here are SYS_ACT: ["OFFER", "NOTIFY_FAILURE"], and USR_ACT: ["AFFIRM"], e.g., accept new proposal
						show_turn(0, prev_turn)
						show_turn(0, sys_turn)
						show_turn(0, turn)
						input("press...")
						stats["AFFIRM"] += 1
					else:
						stats["unknown"] += 1

					# NOTIFY_FAILURE
#					if "INFORM" in acts:
#						stats["INFORM"] += 1
#					elif "AFFIRM" in acts:
#						stats["AFFIRM"] += 1
#					elif "NEGATE" in acts:
#						stats["NEGATE"] += 1
#					else:
#						stats["unknown"] += 1

				prev_turn = turn
	for k, v in stats.items():
		print("{} -> {}".format(k, v))
	print("Total offer turns: {}".format(n))
