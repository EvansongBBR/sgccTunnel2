# Text to SQL (on CSgSQL dataset)

*基于Multilingual-BART的Seq2Seq方法*
## 0.环境准备

Multilingual-BART:
```shell
wget https://dl.fbaipublicfiles.com/fairseq/models/mbart/mbart.cc25.v2.tar.gz 
tar -xzvf mbart.cc25.v2.tar.gz
```
数据集置于 `./data/CSgSQL`目录。

运行环境：
```shell
conda create -n text2sql python=3.8
conda activate text2sql
pip install -r requirements.txt
```

## 1.预处理

执行命令:
```shell
python preprocess.py -d CSgSQL -b /path/to/bart_model [-o OUTPUT_PATH]
```
如果需要dev集也使用正确的表进行测试，使用下面的命令：
```shell
python preprocess.py -d CSgSQL -b /path/to/bart_model -g [-o OUTPUT_PATH]
```
PS. 如果只想测试选表的性能，则执行：
(命令行具体参数见`./sql/select_tables/config.py`)
```shell
python -m sql.select_tables.select_tables
```
选表部分的性能：
```text
recall (table level): 92 %
recall (example level): 83 %
```

## 2.训练
执行命令：
```shell
python train.py \
    --dataset_path ${DATA_OUTPUT_PATH}/bin/ \
    --exp_name ./csgsql \
    --models_path ./models \
    --total_num_update 50000 \
    --max_tokens 1024 \
    --bart_model_path /path/to/bart_model
```
如需修改其他训练参数，请参考`train.py`。

## 3.预测生成
在DEV集上进行预测：
```shell
fairseq-generate ${DATA_OUTPUT_PATH}/bin \
    --path ./models/csgsql/checkpoint_${STEP}.pt \
    --gen-subset valid \
    --nbest 1 \
    --max-tokens 4096 \
    --max-len-b 4096 \
    --source-lang src --target-lang tgt \
    --results-path ./models/csgsql/output_${STEP} \
    --beam 5 \
    --bpe 'sentencepiece' \
    --sentencepiece-model /path/to/bart_model \
    --remove-bpe \
    --skip-invalid-size-inputs-valid-test
```
将会在模型保存目录输出`output_${STEP}/generate-valid.txt`文件。


## 4.后处理+评价

```shell
python process_and_eval.py --path /path/to/generate-valid.txt
```
将会在`generate-valid.txt`所在目录输出：

`generate-valid.txt.eval`（评价文件）

`generate-valid.txt.sql`（最终生成的SQl语句）

并在stdout输出后处理和评价结果。

\
\
最终dev集上的结果：

（dev集随机切分得到，共300条数据，实际数据由于切分方式不同，结果可能存在差异）
```text
Exact Match (without value):
    - 53.2 % (gold tables + top candidate tables = totally 6 tables per example)
    - 36.0 % (top 6 candidate tables)
```