import json, re

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
#	gen = gen.split()
#	print(gen)
	print("*** Dial_id: {} ***".format(dial_id))
	for tag in ["CTX", "SYS_UTT", "SYS_ACT", "GOAL", "SNT", "RA", "GC", "USR_ACT", "USR_UTT"]:
		segment = find_segment(gen, tag)
		if segment is not None:
			print('{} -> "{}"'.format(tag, _color(segment)))
		else:
			print("Fail to find the segment...")
			print("GEN:", gen)
	print("---"*30)
#	input("press...")


def get_original_act_set():
	# full act vocab:
	# https://github.com/ConvLab/ConvLab/blob/master/data/multiwoz/annotation/Multiwoz%20data%20analysis.md#dialog-act
	acts = set()
	acts.add("Inform")
	acts.add("Request")
	acts.add("NoOffer") # equivalent to the concept of `no matching`, `cannot find` in database
	acts.add("Recommend")
	acts.add("Select")
	acts.add("OfferBook") # only for `train` domain, ask if book is needed, equivalent to `Booking-Inform` with [[none, none]] args in restaurant/hotel domain
	acts.add("OfferBooked") # only for `train` domain, inform booking is complete, with corresponding info (such as ref number)
	acts.add("Book") # inform booking is successful, equivalent to `OfferBooked` above
	acts.add("NoBook") # inform booking fails, might because of no availability, usually come together act `request`
	acts.add("bye")
	acts.add("greet")
	acts.add("reqmore")
	acts.add("welcome")
	acts.add("thank")
	return acts


def get_act_natural_language(act):
	if act in ["bye", "greet", "reqmore", "welcome", "thank"]:
		return act

	assert act[0].isupper()
	tokens = re.findall('[A-Z][^A-Z]*', act) # e.g., `FindEvents` -> `Find Events`
	tokens = list(map(str.lower, tokens)) # lower case, -> `find events`
	act_nl = " ".join(tokens)
	return act_nl


def convert_act_into_sgd(act, SPECIAL_TOKENS):
	# TODO: check inference result to see if mapping on NoOffer, OfferBook and NoBook are fine
	"""
		convert multiwoz acts (w/o domain info) into sgd acts ensure that acts with same concept use one name
		e.g., Book (OfferBooked) -> NOTIFY_SUCCESS, NoBook -> NOTIFY_FAILURE
	"""
	if act == "NoOffer":
		act = "NOTIFY_FAILURE"

	elif act == "Recommend":
		act = "OFFER"

	# technically, `OfferBook` is equivalent to (`act=OFFER_INTENT, slot=intent, value=ReserveRestaurant`) on system side in sgd
	# since (1) the conversion is not trivial (completely different representations) and (2) multiwoz has no slot called `intent`
	# one cannot simply convert `OfferBook` to `OFFER_INTENT`
	# we thus keep the act as is
	# note that there is no slot `intent` and value conveying intents in multiwoz
	elif act == "OfferBook":
		act = "Offer_Book"

	elif act == "OfferBooked":
		act = "NOTIFY_SUCCESS"

	elif act == "Book": # same as `OfferBooked`
		act = "NOTIFY_SUCCESS"

	elif act == "NoBook":
		act = "NOTIFY_FAILURE"

	elif act == "bye":
		act = "GOODBYE"

	elif act == "reqmore":
		act = "REQ_MORE"

	elif act == "thank":
		act = "THANK_YOU"
#	elif act == "greet":
#	elif act == "welcome":
	act = act.upper() # align with sgd acts, e.g., `Inform` -> `INFORM`

	# check if valid
	assert "_{}_".format(act) in SPECIAL_TOKENS["additional_special_tokens"]
	return act


def load_schema(schema_file):
	def _update(key, value, mapping):
		if key in mapping:
			assert value == mapping[key] # ensure service meta is the same between data splits
		else:
			mapping[key] = value


	def _restructure_service_meta(service_meta, attribute):
		""""convert slot/intent metadata list into dict(slot/intent=metadata)"""
		assert attribute in ["slots", "intents"]
		mapping = {}
		for value in service_meta[attribute]:
			key = value["name"]
			if attribute == "slots": # domain-slot in multiwoz
				assert "-" in key
				_, key = key.split("-") # domain, slot
				key = normalise_slot(key)
			else: # intent
				key = normalise_intent(key)
			mapping[key] = value
		service_meta[attribute] = mapping

	with open(schema_file) as f:
		data = json.load(f)

	SERVICE2META = {}
	SLOTS, INTENTS = set(), set()
	for service_meta in data:
		service = service_meta["service_name"]
		_restructure_service_meta(service_meta, "slots")
		_restructure_service_meta(service_meta, "intents")
		_update(service, service_meta, SERVICE2META)

		# collect domain-independent slots
#		for domain_slot in service_meta["slots"]:
#			assert "-" in domain_slot
#			domain, slot = domain_slot.split("-")
#			slot = normalise_slot(slot)
#			SLOTS.add(slot)
		for slot in service_meta["slots"]:
			SLOTS.add(slot)

		for intent in service_meta["intents"]:
#			intent = normalise_intent(intent)
			INTENTS.add(intent)

	print("Load schema, intents: {}, slots: {}".format(len(INTENTS), len(SLOTS)))
	return SERVICE2META, INTENTS, SLOTS


def normalise_intent(intent):
	"""convert intent into natural language, e.g., find_hotel -> find hotel"""
	if intent == "police": intent = "find_police"
	if intent == "book_taxi": intent = "find_taxi"
	assert "_" in intent
	return " ".join(intent.split("_"))
		

def normalise_slot(slot):
	if slot == "pricerange":
		return "price range"

	elif slot == "bookday":
		return "book day"

	elif slot == "bookpeople":
		return "book people"

	elif slot == "booktime":
		return "book time"

	elif slot == "bookstay":
		return "book stay"

	elif slot == "ref":
		return "reference"

	elif slot == "arriveby":
		return "arrive by"

	elif slot == "leaveat":
		return "leave at"

	elif slot == "trainid":
		return "train id"

	elif slot == "openhours":
		return "open hours"

	elif slot == "entrancefee":
		return "entrance fee"

	elif slot in ["none", "?"]:
#		return "_Empty_" # special token mark will be added during sequence linearlisation
		return "Empty"

	else:
		return slot


def normalise_value(value):
	# deal with binary and empty values
	if value == "yes":
#		return "_True_"
		return "True"

	elif value == "no":
#		return "_False_"
		return "False"

	elif value in ["none", "?"]:
#		return "_Empty_"
		return "Empty"

#	if value == "swimmingpool": # for simplicity, dont split
#		return "swimming pool"

	else:
		return value


def wrap_element(content_type, content):
	'''
		wrap elements such as slot, value, e.g., <SLOT/> slot </SLOT>
	'''
	assert "/" not in content_type
	return "<{}/> {} </{}>".format(content_type, content, content_type)


def add_str(str1, str2):
	return (str1 + " " + str2)


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
