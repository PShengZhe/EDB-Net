cuda=0
commont=EDB
seed=802
for path in 01 02 03 04 05;
do
    python src_org/main.py \
        --dataset HuffPost \
        --dataFile data/HuffPost/few_shot/${path} \
        --fileVocab=./bert-base-uncased \
        --fileModelConfig=./bert-base-uncased/config.json \
        --fileModel=./bert-base-uncased \
        --fileModelSave result${path} \
        --numKShot 5 \
        --sample 1 \
        --numDevice=$cuda \
        --numFreeze 6 \
        --commont=$commont \
	--seed=$seed


    python src_org/main.py \
        --dataset HuffPost \
        --dataFile data/HuffPost/few_shot/${path} \
        --fileVocab=./bert-base-uncased \
        --fileModelConfig=./bert-base-uncased/config.json \
        --fileModel=./bert-base-uncased \
        --fileModelSave result${path} \
        --numKShot 1 \
        --sample 1 \
        --numDevice=$cuda \
        --numFreeze 6 \
        --commont=$commont \
	--seed=$seed

done
