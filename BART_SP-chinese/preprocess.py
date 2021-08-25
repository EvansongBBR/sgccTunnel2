import argparse
import os
import pickle
import json
import shutil

from sql.build_csgsql_data import build_csgsql_data, read_database_schema
from sql.select_tables_and_process_schema import select_tables_and_process_schema
from sql.spider_utils import disambiguate_items, fix_number_value
from tqdm import tqdm

from train import run_command


def write_interaction(interaction_list, split, output_dir):
    json_split = os.path.join(output_dir, split + '.json')
    pkl_split = os.path.join(output_dir, split + '.pkl')

    rnt = []
    with open(json_split, 'w') as outfile:
        for interaction in interaction_list:
            rnt.append(json.dumps(interaction, ensure_ascii=False))
        outfile.write('\n'.join(rnt))
    new_objs = []
    for i, obj in enumerate(interaction_list):
        sql = obj["sql"]
        sqls = [sql]
        tok_sql_list = []
        for sql in sqls:
            results = []
            tokenized_sql = sql.split()
            tok_sql_list.append((tokenized_sql, results))
        obj["sql"] = tok_sql_list
        new_objs.append(obj)

    with open(pkl_split, 'wb') as outfile:
        pickle.dump(new_objs, outfile)


def sql_tokenize(ex):
    ex['query'] = ex['query'].replace('(', ' ( ').replace(')', ' ) ').replace(',', ' , ') \
        .replace('>', ' > ').replace('<', ' < ').replace('> =', '>=').replace('< =', '<=').strip().strip(';').replace(
        '  ', ' ')
    sql_tokens = ex['query'].split()
    result = []
    result_no_val = []
    values = []
    tmp = ''
    for tok in sql_tokens:
        if not tmp and tok.startswith('\''):
            if not tok.endswith('\''):
                tmp = tok
            else:
                result.append(tok)
                values.append(tok)
                result_no_val.append('value')
        elif tmp:
            tmp += ' ' + tok
            if tmp.endswith('\''):
                result.append(tmp)
                values.append(tmp)
                result_no_val.append('value')
                tmp = 0
        else:
            result.append(tok.lower())
            result_no_val.append(tok.lower())
    return result, result_no_val, values


def read_CSgSQL_split(split_json, database_schemas, dataset):
    entity_map = dict()

    with open(split_json) as f:
        split_data = json.load(f)
    print('read_spider_split', split_json, len(split_data))
    examples = []
    for i, ex in enumerate(tqdm(split_data)):
        # print(i)
        db_id = ex['db_id']
        if db_id not in entity_map:
            entity_map[db_id] = dict()
            for i, tab in enumerate(database_schemas[db_id]['table_names']):
                entity_map[db_id][tab] = i

        skip = False
        query_toks, query_toks_no_value, values = sql_tokenize(ex)
        ex['query_toks'] = query_toks
        ex['query_toks_no_value'] = query_toks_no_value
        try:
            ex = fix_number_value(ex)
        except Exception as e:
            print(ex['query_toks'])
            print(ex['query_toks_no_value'])
            raise e
        entities_map = json.load(open(f'./data/{dataset}/entities_map.json', 'r'))
        try:
            ex['query_toks_no_value'] = disambiguate_items(db_id, ex['query_toks_no_value'],
                                                           tables_file=f'./data/{dataset}/db_schema.json',
                                                           allow_aliases=False)
            k = 0
            for i, val in enumerate(ex['query_toks_no_value']):
                if val == "'value'":
                    ex['query_toks_no_value'][i] = values[k]
                    k += 1
                if val in entities_map:
                    ex['query_toks_no_value'][i] = entities_map[val]

        except Exception as e:
            print(f'ERROR {i}:', query_toks)
            print(f'ERROR {i}:', ex['query'])
            print(e)
            raise e

        final_sql = ' '.join(ex['query_toks_no_value'])

        if skip and 'train' in split_json:
            print('skip', final_sql)
            continue

        final_sql_parse = final_sql

        final_utterance = ex['question']
        item = {'database_id': db_id,
                'question_id': ex['question_id'],
                'utterance': final_utterance,
                'sql': final_sql_parse,
                'selected_tables': ex['selected_tables'],
                }
        examples.append(item)
    return examples


def read_CSgSQL(spider_dir, database_schemas, dataset):

    train_json = os.path.join(spider_dir, 'train_new.json')
    train_data = read_CSgSQL_split(train_json, database_schemas, dataset)

    dev_json = os.path.join(spider_dir, 'dev_new.json')
    dev_data = read_CSgSQL_split(dev_json, database_schemas, dataset)

    return train_data, dev_data


def preprocess(dataset, use_gold_tables, bart_model_path, output_data_path=None):
    print('selecting candidate tables and tagging questions...')
    select_tables_and_process_schema(dataset, use_gold_tables=use_gold_tables)
    CSgSQL_dir = f'data/{dataset}/'
    database_schema_filename = f'data/{dataset}/db_schema_new.json'
    # output_dir = 'data/CSgSQL_data_new_sl2'
    output_dir = f'data/{dataset}_sl'

    if os.path.isdir(output_dir):
        shutil.rmtree(output_dir)
    os.mkdir(output_dir)

    schema_tokens = {}
    column_names = {}
    database_schemas = {}

    print('Reading database schema file...')
    schema_tokens, column_names, database_schemas = read_database_schema(database_schema_filename, schema_tokens,
                                                                         column_names, database_schemas)

    output_database_schema_filename = os.path.join(output_dir, 'tables.json')
    with open(output_database_schema_filename, 'w') as outfile:
        json.dump([v for k, v in database_schemas.items()], outfile, indent=4)

    train_data, dev_data = read_CSgSQL(CSgSQL_dir, database_schemas, dataset)

    print('train examples: ', len(train_data))
    print('dev examples: ', len(dev_data))

    write_interaction(train_data, 'train', output_dir)
    write_interaction(dev_data, 'dev', output_dir)
    output_data_path = f"./dataset_post/{dataset}_vanilla" if not output_data_path else output_data_path
    print('building data...')
    build_csgsql_data(overnight_path=output_dir,
                      out_path=output_data_path,
                      spm_model_path=os.path.join(bart_model_path, 'sentence.bpe.model'))
    dict_path = os.path.join(bart_model_path, 'dict.txt')
    cmd = f'''
    fairseq-preprocess \
   --source-lang src \
   --target-lang tgt \
   --trainpref {output_data_path}/train.spm \
   --validpref {output_data_path}/dev.spm \
   --destdir {output_data_path}/bin \
   --thresholdtgt 0 \
   --thresholdsrc 0 \
   --srcdict {dict_path} \
   --tgtdict {dict_path} \
   --workers 1
    '''
    print("running `fairseq-preprocess` ...")
    run_command(cmd)
    print('Preprocess done!')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", '-d', type=str, default='CSgSQL')
    parser.add_argument("--use-gold-tables", '-g', action='store_true', default=False)
    parser.add_argument("--bart-model-path", '-b', type=str, default='./data/ptm/')
    parser.add_argument("--output-path", '-o', type=str, default=None)
    args = parser.parse_args()
    preprocess(args.dataset, args.use_gold_tables, args.bart_model_path, args.output_path)
