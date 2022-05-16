# gpt2-user-model

Note: this model is still under development and has not been empirically evaluated.

The model has been only tested using Alex's evaluation script and show superiority over user simulators in Convlab in corresponding metrics.


## Set up environment
```console
>> python3.7 -m venv env
>> pip install -r requirements.txt
```

## Pre-process data
### SGD
```console
>> python utils/preprocess_sgd.py SGD_data_path
```
`SGD_data_path`: data path to the original SGD dataset

### MultiWOZ
```console
>> python utils/preprocess_multiwoz.py MultiWOZ_data_path
```
`MultiWOZ_data_path`: data path to the original MultiWOZ v2.2 dataset

The processed data will be stored in `processed_data/sgd` and `processed_data/multiwoz` folders.


## Training
```console
>> bash train.sh dataset
```
`dataset`: specified dataset for model training. Options are `SGD`, `MultiWOZ` or `Joint`. Use `Joint` to train on an aggregated dataset.

Model checkpoints will be stored at `checkpoint` folder.


## Decoding
```console
>> bash decode.sh dataset model_checkpoint_path
```
`model_checkpoint_path`: the checkpoint path to a trained model


## Interaction (MultiWOZ model only)
Run the following command to interact with the trained user model

```console
>> cd interaction/
>> python multiwoz_interact.py model_checkpoint_path
```
