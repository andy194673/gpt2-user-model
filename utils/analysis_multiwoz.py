import os, sys
import json


DATA_SPLIT = ['train', 'dev', 'test']


def _check_n_turns(data, data_act):
	for split in DATA_SPLIT:
		for dial_id, meta in data[split].items():
			n_in_meta = len(meta["turns"])

			assert dial_id in data_act
			n_in_act = len(data_act[dial_id])
			assert n_in_meta == n_in_act


def collect_data(data_path, remove_dial_switch=False):
	# load act
	act_file = os.path.join(data_path, "dialog_acts.json")
	with open(act_file) as f:
		data_act = json.load(f)
	print("Load {} dialogues in act file".format(len(data_act)))

	# load data
	data = {}
	for split in DATA_SPLIT:
		data[split] = iter_data_folder(data_path, split, remove_dial_switch, data_act)

	_check_n_turns(data, data_act)
	return data, data_act


def remove_dial(dial_id, dial, dial_act):
	# check services
	services = dial["services"]
	if "police" in services or "bus" in services or "hospital" in services:
		return True

	# check act
	domains = set()
	for turn_id, turn_act in dial_act.items():
		dialogue_act = turn_act["dialog_act"]
		for dact in dialogue_act:
			assert "-" in dact
			domain, act = dact.split("-")
			domains.add(domain)
	if "Police" in domains or "Bus" in domains or "Hospital" in domains:
		return True
	return False


def iter_data_folder(data_path, split, remove_dial_switch, data_act):
	''' Iterate data folder '''
	split_dir = os.path.join(data_path, split)
	data_split = {}
	remove_dial_ids = []
	total_dial_ids = []
	for f in os.listdir(split_dir):
		if not f.startswith('dialogues'): # skip schema.json
			continue
		file_path = os.path.join(data_path, split, f)
		iter_file(file_path, data_split, remove_dial_ids, total_dial_ids, remove_dial_switch, data_act)
	print('Done collecting {} | total {} dialogues | load {} dialogues | remove {} dialogues'.format(split, len(total_dial_ids), len(data_split), len(remove_dial_ids)))
	return data_split


def iter_file(file_path, data_split, remove_dial_ids, total_dial_ids, remove_dial_switch, data_act):
	with open(file_path) as f:
		data_in = json.load(f) # list of dialouges in a json file

	for dial in data_in:
		dial_id = dial['dialogue_id']
		total_dial_ids.append(dial_id)
		dial_act = data_act[dial_id]

		if remove_dial_switch and remove_dial(dial_id, dial, dial_act):
			remove_dial_ids.append(dial_id)
		else:
			data_split[dial_id] = dial


def show_dial(dial_id, data, data_act):
	def simple_linearise_act(dialouge_act):
		linear_act = ""
		for domain_act, slot_value_list in dialouge_act.items():
			linear_act += (domain_act + " ")
			for slot_value in slot_value_list:
				slot, value = slot_value[0], slot_value[1]
				linear_act += (slot + " ")
				linear_act += (value + " ")
		return linear_act

	split = None
	for data_split in DATA_SPLIT:
		if dial_id in data[data_split]:
			split = data_split
			break

	print("dial_id: {}".format(dial_id))
	for turn_id, turn in enumerate(data[split][dial_id]["turns"]):
		dialouge_act = data_act[dial_id][str(turn_id)]["dialog_act"]
		linear_act = simple_linearise_act(dialouge_act)
		print("-----"*15)
		print("turn_id: {}, spk: {}".format(turn_id, turn["speaker"]))
		print("act: |{}|".format(linear_act))
		print("utt: |{}|".format(turn["utterance"]))
