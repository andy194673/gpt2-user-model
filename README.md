# gpt2-user-model

Although the model seems was  using Alex's 

Note: this model is still under development and has not been empirically evaluated.

The model has been only tested using Alex's evaluation script and show superiority over user simulators in Convlab.


## Set up environment
```console
>> python3.7 -m venv env
>> pip install -r requirements.txt
```

## Pre-process data
```console
>> python utils/preprocess_sgd.py SGD-data-path
>> python utils/preprocess_multiwoz.py MultiWOZ-data-path
```

`SGD-data-path`: data path to the original SGD dataset

`MultiWOZ-data-path`: data path to the original MultiWOZ v2.2 dataset

The processed data will be stored in `processed_data/sgd` and `processed_data/multiwoz` folders
