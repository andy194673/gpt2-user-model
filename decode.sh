experiment=$1
checkpoint=$2

if [[ "$experiment" == "SGD" ]]; then
	echo "Conduct experiment with SGD dataset"
	job_name='SGD-full'
	data_list="sgd" # 165k training examples
	eval_interval=50000 # evaluation interval

elif [[ "$experiment" == "MultiWOZ" ]]; then
	echo "Conduct experiment with MulwiWOZ dataset"
	job_name='MultiWOZ-full'
	data_list="multiwoz" # 56k training examples
	eval_interval=20000

elif [[ "$experiment" == "Joint" ]]; then
	echo "Conduct experiment with SGD + MulwiWOZ dataset"
	job_name='Joint-full'
	data_list="sgd multiwoz" # 221k training examples
	eval_interval=70000

else
	echo "Unrecognised argument"
	exit
fi

mkdir -p log decode
decode_file='decode/'$job_name'.json'
eye_browse_output=true # set to false for storing generation results in file 

python main.py  --mode='testing' \
				--model_name=$job_name \
				--checkpoint=$checkpoint \
				--decode_file=$decode_file \
				--data_dir="processed_data" \
				--data_list=$data_list \
				--eye_browse_output=$eye_browse_output
