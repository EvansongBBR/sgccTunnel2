import re
from re import RegexFlag
import argparse
from semparse.worlds.evaluate_spider import evaluate as evaluate_sql
from tqdm import tqdm
from sql.datasets.spider import *
import nltk
nltk.download('punkt')

def post_processing_sql(p_sql, foreign_key_maps, schemas, o_schemas, source):
    foreign_key = {}
    for k,v in foreign_key_maps.items():
        if k==v:
            continue
        key=' '.join(sorted([k.split('.')[0].strip('_'),v.split('.')[0].strip('_')]))
        foreign_key[key]=(k.strip('_').replace('.','@'), v.strip('_').replace('.','@'))

    primary_key = {}
    for t in o_schemas.tables:
        table = t.orig_name.lower()
        if len(t.primary_keys)==0:
            continue
        column = t.primary_keys[0].orig_name.lower()
        primary_key[table]=f'{table}@{column}'

    p_sql=p_sql.split()

    columns = ['*']
    tables = []
    for table, column_list in schemas.schema.items():
        for column in column_list:
            columns.append(f'{table}@{column}')
        tables.append(table)

    ##correct column
    for idx,token in enumerate(p_sql):
        if '@' in token and token not in columns:
            for item in columns:
                if item.split('@')[-1]==token.split('@')[-1]:
                    p_sql[idx]=item
                    print(f'column\t{token} -> {item}')
                    break
                elif item.split('@')[0]==token.split('@')[0]:
                    p_sql[idx]=item
                    print(f'column\t{token} -> {item}')
                    break

    ##correct join clause
    for idx,token in enumerate(p_sql):
        if token=='from':
            table_collect = []
        if token == 'join':
            if '@' in p_sql[idx-1]:
                right_table = p_sql[idx + 1]
                table_collect.append(right_table)

                for p_table in table_collect:
                    key = ' '.join(sorted([p_table, table_collect[-2]]))
                    if key in foreign_key:
                        continue
            else:
                left_table = p_sql[idx-1]
                right_table = p_sql[idx+1]
                table_collect.append(left_table)
                table_collect.append(right_table)
                key = ' '.join(sorted([left_table, right_table]))
        elif token == 'on':
            #foreign keys
            if key in foreign_key:
                p_sql[idx+1], p_sql[idx+3]= foreign_key[key]
                print(f'on clause\t{foreign_key[key]}')
            else:
                try:
                    p_sql[idx+1]=primary_key[key.split(' ')[0]]
                    p_sql[idx+3] = primary_key[key.split(' ')[-1]]
                except:
                    pass

    return ' '.join(p_sql)


def extract_structure_data(plain_text_content: str, file_path):
    def sort_by_id(data):
        data.sort(key=lambda x: int(x.split('\t')[0][2:]))
        return data

    # extracts lines starts with specific flags
    # map id to its related information
    data = []

    original_schemas = load_original_schemas('./data/CSgSQL/db_schema_new.json')
    schemas, eval_foreign_key_maps = load_tables('./data/CSgSQL/db_schema_new.json')

    sql_to_db = {}
    with open(file_path, "r") as data_file:
        example_lines = data_file.readlines()
    for line in tqdm(example_lines):
        item = json.loads(line.strip())
        db_id = item['database_id']
        ground_str = item['sql'].lower()
        if ground_str in sql_to_db and sql_to_db[ground_str] != db_id:
            print(ground_str, db_id)
        sql_to_db[ground_str] = db_id

    predict_outputs = sort_by_id(re.findall("^D.+", plain_text_content, RegexFlag.MULTILINE))
    ground_outputs = sort_by_id(re.findall("^T.+", plain_text_content, RegexFlag.MULTILINE))
    source_inputs = sort_by_id(re.findall("^S.+", plain_text_content, RegexFlag.MULTILINE))

    assert len(predict_outputs)==len(ground_outputs)==len(source_inputs)
    for predict, ground, source in zip(predict_outputs, ground_outputs, source_inputs):
        try:
            predict_id, predict_score, predict_clean = predict.split('\t')
            ground_id, ground_clean = ground.split('\t')
            source_id, source_clean = source.split('\t')
            assert predict_id[2:] == ground_id[2:]
            assert ground_id[2:] == source_id[2:]
        except Exception:
            print("An error occurred in source: {}".format(source))
            continue

        if ground_clean == 'PAD':
            continue

        db_id = 'AI_SEARCH'
        predict_clean = predict_clean
        predict_clean = post_processing_sql(predict_clean, eval_foreign_key_maps[db_id], original_schemas[db_id],
                                            schemas[db_id], source)
        data.append((predict_score, predict_id[2:], predict_clean, ground_clean, source_clean.split('<Q>')[-1].strip()))

    return data


def evaluate(file_path, data: List):

    def evaluate_example(_predict_str: str, _ground_str: str):
        return re.sub("\s+", "", _predict_str.lower()) == re.sub("\s+", "", _ground_str.lower())

    correct_num = 0
    correct_arr = []
    total = len(data)

    sql_to_db={}
    with open(file_path, "r") as data_file:
        example_lines = data_file.readlines()
    for line in tqdm(example_lines):
        item = json.loads(line.strip())
        db_id=item['database_id']
        ground_str=item['sql'].lower()
        if ground_str in sql_to_db and sql_to_db[ground_str]!=db_id:
            print(ground_str,db_id)
        sql_to_db[ground_str]=db_id
    db_id = sql_to_db[ground_str]

    for example in data:

        _, _, predict_str, ground_str, source_str = example
        db_id = 'AI_SEARCH'
        sql_match = evaluate_sql(gold=ground_str.replace('@','.'),
                                 predict=predict_str.replace('@','.'),
                                 db_name=db_id,
                                 db_dir='./data/CSgSQL/db_content_new.json',
                                 table='./data/CSgSQL/db_schema_new.json')
        if sql_match or evaluate_example(predict_str, ground_str):
            is_correct = True
        else:
            is_correct = False
        if is_correct:
            correct_num += 1
        correct_arr.append(is_correct)

    print("Correct/Total : {}/{}, {:.3f}".format(correct_num, total, correct_num / total))
    return correct_arr, correct_num, total


def process_file(generate_file_path, dataset_type):
    with open(generate_file_path, "r", encoding="utf8") as generate_f:
        file_content = generate_f.read()
        file_path = f'./data/{dataset_type}_sl/dev.json'
        data = extract_structure_data(file_content, file_path)
        correct_arr, correct_num, total = evaluate(file_path, data)
        # write into eval file
        eval_file_path = generate_file_path + ".eval"
        eval_file = open(eval_file_path, "w", encoding="utf8")
        eval_file.write("Eval\tScore\tID\tPredict\tGolden\tSource\n")
        for example, correct in zip(data, correct_arr):
            eval_file.write(str(correct) + "\n" + "\n".join(example) + "\n\n")
        eval_file.close()
        entities_map = json.load(open('./data/CSgSQL/entities_map.json'))
        entities_map_r = {v:k for k, v in entities_map.items()}
        final_output_file = open(generate_file_path + '.sql', 'w')
        for _, _, predict_str, _, _ in data:
            pred_toks = predict_str.split()
            for i, tok in enumerate(pred_toks):
                if tok in entities_map_r:
                    pred_toks[i] = entities_map_r[tok]
            final_output_file.write(' '.join(pred_toks) + '\n')
    return correct_num, total


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", default="/mnt/d/generate-valid.txt")
    parser.add_argument("--dataset_type", default="CSgSQL")

    args = parser.parse_args()

    process_file(args.path, args.dataset_type)
