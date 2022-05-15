# gpt2-user-model

## Set up environment
```console
>> python3.7 -m venv env
>> pip install -r requirements.txt
```

## Pre-process data
```console
>> python utils/preprocess_sgd.py SGD-data-path
```
`SGD-data-path`: the data path to original the SGD dataset
The processed data will be stored at `processed_data/sgd` folder
