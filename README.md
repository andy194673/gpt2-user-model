# gpt2-user-model
An end-to-end GPT-2 based user simulator for task-oriented dialogue, developed on both the [MultiWOZ dataset](https://github.com/budzianowski/multiwoz) and [SGD dataset](https://github.com/google-research-datasets/dstc8-schema-guided-dialogue).

Note: this model is still under development and has not been empirically evaluated.

The model has been only tested internally using [CGDF1 evaluation script](https://github.com/alexcoca/gcdf1), a user simulator evalator at semantic level, and showed superiority over user simulators in Convlab in corresponding metrics.


## Set up environment
```console
>> python3.7 -m venv env
>> pip install -r requirements.txt
```

## Pre-process data
### SGD
```console
>> python utils/preprocess_sgd.py $SGD_data_path
```
`SGD_data_path`: data path to the original SGD dataset

### MultiWOZ
```console
>> python utils/preprocess_multiwoz.py $MultiWOZ_data_path
```
`MultiWOZ_data_path`: data path to the original MultiWOZ v2.2 dataset

The processed data will be stored in `processed_data/sgd` and `processed_data/multiwoz` folders.


## Training
```console
>> bash train.sh $dataset
```
`dataset`: specified dataset for model training. Options are `SGD`, `MultiWOZ` or `Joint`. Use `Joint` to train on an aggregated dataset.

Model checkpoints will be stored at `checkpoint` folder.


## Decoding
```console
>> bash decode.sh $dataset $model_checkpoint_path
```
`model_checkpoint_path`: the checkpoint path to a trained model


## Download a trained model
We provide a user model trained on MultiWOZ. Run the following command to download the model checkpoint `MultiWOZ-full_checkpoint_step340k`. 
```console
>> wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1jRR-YYDyPORzmmyANecUjmLKciRSlM4l' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1jRR-YYDyPORzmmyANecUjmLKciRSlM4l" -O model.tar.gz && rm -rf /tmp/cookies.txt
>> tar zxvf model.tar.gz
```


## Interaction (MultiWOZ model only)
Run the following command to interact with the trained user model where you play the dialogue agent and provide the system response.

```console
>> cd interaction/
>> python multiwoz_interact.py $model_checkpoint_path
```
Example dialogue through interaction:

![alt text](https://github.com/andy194673/gpt2-user-model/blob/main/.images/example-1.png)
![alt text](https://github.com/andy194673/gpt2-user-model/blob/main/.images/example-2.png)
![alt text](https://github.com/andy194673/gpt2-user-model/blob/main/.images/example-3.png)
![alt text](https://github.com/andy194673/gpt2-user-model/blob/main/.images/example-4.png)
![alt text](https://github.com/andy194673/gpt2-user-model/blob/main/.images/example-5.png)
