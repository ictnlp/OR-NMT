set -e
set -x

full_path=$(realpath $0)
root=$(dirname $full_path)
export LC_ALL="en_US.UTF-8"
export PYTHONPATH=$PYTHONPATH:$root

#data_bin_name=wmt15_fr2en_bpe${bpe_num}k_bert
#data_bin_name=wmt15_fr_en_bpe${bpe_num}k_fp16
#data_bin_name=wmt15_fr_en_bpe${bpe_num}k_fp16_dlcl
#data_bin_name=wmt20_fr2en_${bpe_num}kbpe_bt43M_fp16
#data_bin_name=fr2en_${bpe_num}kbpe_1to0_filter_fp16
#data_bin_name=fr2en_${bpe_num}kbpe_bt43M1to1_filter_fp16
#data_bin_name=fr2en_${bpe_num}kbpe_bt43M1to2_filter_fp16
Usage() {
    echo "Usage: $0 train|infer data_bin_name exp_name batch_size accumulate">&2
    exit 1
}

[ $# -eq 5 ] || Usage
[ $1 == "train" ] || [ $1 == "infer" ] || Usage

src_lang=fr
trg_lang=en
option=$1
data_bin_name=$2
exp_name=$3
batch_size=$4
accum=$5

work_dir=$root
#data_dir=/ceph_nmt/wenzhang/1.research/mt/diversity/data/$data_bin_name
data_dir=/ceph_nmt/wenzhang/1.research/mt/wmt2020/data/$data_bin_name
model_dir=$work_dir/ckpt/$data_bin_name/$exp_name
mkdir -p $model_dir

if [ ! -z $_PYTHON ]; then
    python=$_PYTHON
else
    python=python
fi

env_cuda=$($python -c "import torch;import os;print(os.environ.get('CUDA_VISIBLE_DEVICES', 'UNK'))")
torch_cuda=$($python -c "import torch;print(torch.cuda.device_count())")
torch_v=$($python -c "import torch; print(torch.__version__)")
cuda_available=$($python -c "import torch;print(torch.cuda.is_available())")

echo -e "\n\t start "${option}".................." >> $model_dir/log 
echo 'show GPU' >> $model_dir/log
nvidia-smi >> $model_dir/log
nvidia-smi -L >> $model_dir/log
date >> $model_dir/log
pwd >> $model_dir/log
echo pyTorch version=$($python -c "import torch; print(torch.__version__)") >> $model_dir/log
echo torch.cuda.device_count=$($python -c "import torch;print(torch.cuda.device_count())") >> $model_dir/log
echo torch.cuda.is_available=$($python -c "import torch;print(torch.cuda.is_available())") >> $model_dir/log

echo data_bin_name=$data_bin_name >> $model_dir/log
echo exp_name=$exp_name >> $model_dir/log
echo batch_size=$batch_size >> $model_dir/log
echo accum_upd=$accum >> $model_dir/log
echo root=$root >> $model_dir/log
echo work_dir=$work_dir >> $model_dir/log
echo data_dir=$data_dir >> $model_dir/log
echo model_dir=$model_dir >> $model_dir/log

export PYTHONWARNINGS='ignore:semaphore_tracker:UserWarning'
if [ $option == "train" ]; then

    #export CUDA_VISIBLE_DEVICES=4,5,6,7
    export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
    #export CUDA_VISIBLE_DEVICES=0,1,2,3
    n_gpus=${CUDA_VISIBLE_DEVICES//,/}
    echo CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES >> $model_dir/log
    echo n_gpus=${#n_gpus} >> $model_dir/log
    a=$(nvidia-smi -L)
    if [[ $a == *"V100"* ]]
    then
        fp='--fp16 --memory-efficient-fp16'
    else
        fp=''
    fi
    fp='--fp16 --memory-efficient-fp16'
    echo fp=$fp

    warmup=''
    finetune=''
    if ( true )
    then
        exps_dir=/ceph_nmt/wenzhang/1.research/mt/wmt2020/fr2en
        pretrained_nmt_model=$exps_dir/mainline/bt2_filter_raw_ftbt_8u8_bigbatch/ckpt/j40k_bt2_filter_raw_ftbt_fp16/big_10240_8u8
        #echo pretrained_nmt_model=$pretrained_nmt_model/checkpoint_avg5_20200422_010445.pt >> $model_dir/log
        #if [ ! -f $model_dir/checkpoint_nmt.pt ]
        #then
        #    cp ${pretrained_nmt_model} $model_dir/checkpoint_nmt.pt
        #fi
        #if [ ! -f "$model_dir/checkpoint_last.pt" ]
        #then
            # 41.13 on newtest2014
            #cp ${pretrained_nmt_model}/checkpoint_best.pt $model_dir/checkpoint_last.pt
            # 41.16 on newtest2014
            #cp ${pretrained_nmt_model}/checkpoint_avg5_20200422_010445.pt $model_dir/checkpoint_last.pt
        #fi
        #finetune="--restore-file ${pretrained_nmt_model}/checkpoint_best.pt"
        finetune="--restore-file ${pretrained_nmt_model}/checkpoint_avg5_20200422_010445.pt"
        warmup='--reset-optimizer'
        #warmup='--reset-lr-scheduler --reset-optimizer'
    fi
    echo finetune=$finetune
    echo warmup=$warmup

    cd $work_dir
    #python setup.py build_ext --inplace
    #pip install -r requirements.txt
    #python setup.py develop
    #python setup.py install
    pip install --editable .
    python -m pip install nltk
    echo 'pip install finish ...'

    $python train.py $data_dir \
        --arch oracle_transformer_vaswani_wmt_en_fr_big --share-all-embeddings \
        --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 --lr-scheduler inverse_sqrt \
        --warmup-init-lr 1e-07 --warmup-updates 4000 --lr 0.0007 --min-lr 1e-09 --dropout 0.1 \
        --weight-decay 0.0 --criterion oracle_label_smoothed_cross_entropy --label-smoothing 0.1 \
        --max-tokens $batch_size --update-freq $accum --no-progress-bar --log-format json --max-update 250000 \
        --log-interval 10 --save-interval 1 --keep-last-epochs 10 \
        --seed 1111 $finetune $warmup \
        --distributed-port 27777 --distributed-world-size ${#n_gpus} --ddp-backend=no_c10d $fp \
        --source-lang ${src_lang} --target-lang ${trg_lang} --save-dir $model_dir \
        --use-word-level-oracles --decay-k 25000 --use-greed-gumbel-noise --gumbel-noise 0.5
        #--source-lang fr --target-lang en --save-dir $model_dir >> $model_dir/log 2>&1
        #--log-interval 10 --save-interval-updates 10000 --keep-interval-updates 10 --save-interval 10000 \
        #--seed 1111 --skip-invalid-size-inputs-valid-test --reset-lr-scheduler --reset-optimizer \
        #--distributed-port 21111 --distributed-world-size 8 --ddp-backend=no_c10d $fp \
        #--distributed-world-size ${#n_gpus} --ddp-backend=no_c10d $fp \
else
    pip install --editable .
    python -m pip install nltk
    #model_dir=$work_dir/ckpt/fr_en_0.5
    export CUDA_VISIBLE_DEVICES=0
    bpe_num=40
    beam_size=4
    batch_size=128
    lenpen=0.6
    buffer_size=4096
    cur_time=$(date "+%Y%m%d_%H%M%S")
    trans_dir=$model_dir/Trans_b${beam_size}_l${lenpen}_${cur_time}
    mkdir -p $trans_dir

    model_name=checkpoint_best.pt
    #testsets=('test' 'test1')
    #fnames=('newstest2014' 'newstest2015')
    #testsets=('test' 'test1' 'test2' 'test3' 'valid' 'test4')
    #fnames=('mt02' 'mt03' 'mt04' 'mt05' 'mt06' 'mt08')
    testsets=('test' 'test1' 'test2')
    fnames=('test' 'newstest2014' 'newstest2015')

    fairseq_dir=/ceph_nmt/wenzhang/1.research/mt/wmt2020/fr2en/fairseq
    moses_script_dir=/ceph_nmt/wenzhang/3.tools/mosesdecoder/scripts
    #devtest_dir=/ceph_nmt/wenzhang/2.data/wmt2020/fr2en/j${bpe_num}k_bt43M/mktest
    devtest_dir=/ceph_nmt/wenzhang/2.data/wmt2020/fr2en/lifelong_fr2en_IWSLT16_TED
    scripts_dir=/ceph_nmt/wenzhang/3.tools/scripts
    cd $trans_dir

    for(( i=0;i<${#testsets[@]};i++ )) do
        testset=${testsets[i]}
        fname=${fnames[i]}
        prefix=${fname}.b${beam_size}.l${lenpen}.best.${trg_lang}
        #python $work_dir/generate.py $data_dir --path $model_dir/$model_name \
        #        --gen-subset $testset --beam $beam_size --batch-size $batch_size --remove-bpe --lenpen $lenpen \
        #        --source-lang fr --target-lang en > ${prefix}
        #python $work_dir/interactive.py $data_dir --path $model_dir/$model_name \
        python ${work_dir}/interactive.py $data_dir --path $model_dir/$model_name --num-workers 4 \
            --buffer-size $buffer_size --beam $beam_size --batch-size $batch_size --remove-bpe --lenpen $lenpen \
            < ${devtest_dir}/${fname}.${src_lang} > ${prefix}

        # because fairseq's output is unordered, we need to recover its order
        grep ^H ${prefix} | cut -f1,3- | cut -c3- | sort -k1n | cut -f2- > ${prefix}.proc
        # no sort for dlcl
        #grep ^H ${prefix} | cut -f1,3- | cut -c3- | cut -f2- > ${prefix}.proc

        # tokenized multi_bleu, the references should be tokenized
        python ${scripts_dir}/multi_bleu.py -c ${prefix}.proc -r ${devtest_dir}/${fname}.${trg_lang} -nr 1 -refbpe

        perl ${moses_script_dir}/tokenizer/detokenizer.perl -l ${trg_lang} < ${prefix}.proc > ${prefix}.proc.detok
        python ${work_dir}/score.py -s ${prefix}.proc.detok -r ${devtest_dir}/${fname}.raw.${trg_lang} --sacrebleu
    done

    #exit 0
    # average last 10 checkpoints
    for avg_num in 5 10; do
        model_name=checkpoint_avg${avg_num}_${cur_time}.pt

        python $work_dir/scripts/average_checkpoints.py --inputs $model_dir \
            --output $model_dir/$model_name \
            --num-epoch-checkpoints $avg_num
            #--num-update-checkpoints $avg_num

        for(( i=0;i<${#testsets[@]};i++ )) do
            testset=${testsets[i]}
            fname=${fnames[i]}
            prefix=${fname}.b${beam_size}.l${lenpen}.avg${avg_num}.${trg_lang}
            #python $work_dir/generate.py $data_dir --path $model_dir/$model_name \
            #        --gen-subset $testset --beam $beam_size --batch-size $batch_size --remove-bpe --lenpen $lenpen \
            #        --source-lang fr --target-lang en > ${prefix}
            #python $work_dir/interactive.py $data_dir --path $model_dir/$model_name \
            python $work_dir/interactive.py $data_dir --path $model_dir/$model_name --num-workers 4 \
                --buffer-size $buffer_size --beam $beam_size --batch-size $batch_size --remove-bpe --lenpen $lenpen \
                < ${devtest_dir}/${fname}.${src_lang} > ${prefix}

            # because fairseq's output is unordered, we need to recover its order
            grep ^H ${prefix}| cut -f1,3- | cut -c3- | sort -k1n | cut -f2- > ${prefix}.proc
            # no sort for dlcl
            #grep ^H ${prefix}| cut -f1,3- | cut -c3- | cut -f2- > ${prefix}.proc

            # tokenized multi_bleu, the references should be tokenized
            python ${scripts_dir}/multi_bleu.py -c ${prefix}.proc -r ${devtest_dir}/${fname}.${trg_lang} -nr 1 -refbpe

            perl ${moses_script_dir}/tokenizer/detokenizer.perl -l ${trg_lang} < ${prefix}.proc > ${prefix}.proc.detok
            python $work_dir/score.py -s ${prefix}.proc.detok -r ${devtest_dir}/${fname}.raw.${trg_lang} --sacrebleu
        done
    done
fi
