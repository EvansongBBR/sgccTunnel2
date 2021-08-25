import copy

import json
from collections import defaultdict
from sql.select_tables import get_candidate_tables


def find_all_tabs(sql_dict, found):
    if isinstance(sql_dict, list):
        for x in sql_dict:
            find_all_tabs(x, found)
    elif isinstance(sql_dict, dict):
        if 'table_units' in sql_dict:
            for t, val in sql_dict['table_units']:
                if t == 'table_unit':
                    found.append(val)
        else:
            for k, v in sql_dict.items():
                find_all_tabs(v, found)


def select_table(gold_tables, candidate_tables, num):
    selected_tables = copy.copy(gold_tables)
    num = num - len(selected_tables)
    i = 0
    candidate_len = len(candidate_tables)
    while num > 0 and i < candidate_len:
        if candidate_tables[i] not in selected_tables:
            selected_tables.append(candidate_tables[i])
            num -= 1
        i += 1
    selected_tables.sort()
    return selected_tables


def select_tables_and_process_schema(dataset, use_gold_tables=False):
    entities_map = {}
    # all_use_gold_tables = True
    with open(f'./data/{dataset}/db_schema.json', 'r') as f:
        db = json.load(f)
        for i, (t, c) in enumerate(db[0]['column_names']):
            db[0]['column_names'][i][1] = c.replace('(', '（').replace(')', '）').replace(' ', '_')

        for t, t1 in zip(db[0]['table_names'], db[0]['table_names_original']):
            entities_map[t1] = t
        for (t_id, c), (_, c1) in zip(db[0]['column_names'], db[0]['column_names_original']):
            if t_id == -1:
                entities_map[c1] = c
            else:
                t_name = db[0]['table_names'][t_id]
                t_name1 = db[0]['table_names_original'][t_id]
                entities_map[t_name1 + '@' + c1] = t_name + '@' + c
        db[0]['table_names_original'] = copy.deepcopy(db[0]['table_names'])
        db[0]['column_names_original'] = copy.deepcopy(db[0]['column_names'])

        with open(f'./data/{dataset}/db_schema_new.json', 'w') as ff:
            json.dump(db, ff, ensure_ascii=False, indent=2)

    with open(f'./data/{dataset}/db_content.json', 'r') as f:
        db = json.load(f)
        tables = {}
        for k, v in db[0]['tables'].items():
            for i, c in enumerate(v['header']):
                v['header'][i] = entities_map[k + '@' + c].split('@')[-1]
            tables[entities_map[k]] = v
        db[0]['tables'] = tables
        with open(f'./data/{dataset}/db_content_new.json', 'w') as ff:
            json.dump(db, ff, ensure_ascii=False, indent=2)

    tables_dict = json.load(open(f'./data/{dataset}/db_schema.json', 'r'))

    tables = defaultdict(lambda: [None, list(), None])
    for t_id, column in tables_dict[0]['column_names']:
        if t_id == -1:
            continue
        tab = tables_dict[0]['table_names'][t_id]
        if tables[tab][0] is None:
            tables[tab][0] = tables_dict[0]['table_names_original'][t_id]
            tables[tab][2] = t_id
        tables[tab][1].append(column)

    csgsql_data = {
        'train': json.load(open(f'./data/{dataset}/train.json', 'r')),
        'dev': json.load(open(f'./data/{dataset}/dev.json', 'r'))
    }

    for section, data in csgsql_data.items():
        all_candidate_tables = get_candidate_tables(
            predict_data_path=f'./data/{dataset}/{section}.json',
            schema_path=f'./data/{dataset}/db_schema.json',
            content_path=f'./data/{dataset}/db_content.json',
            answer_size=10
        )
        print(len(all_candidate_tables))
        print(len(data))
        assert len(all_candidate_tables) == len(data)
        for item, candidate_tables in zip(data, all_candidate_tables):
            selected_entities = []
            q = item['question'].lower()
            if section == 'train' or use_gold_tables:
                gold_tables = []
                find_all_tabs(item['sql'], gold_tables)
                selected_tables = select_table(gold_tables,
                                               candidate_tables['predict_tables_id'],
                                               num=6)
            else:
                selected_tables = candidate_tables['predict_tables_id'][:6]

            item['selected_tables'] = selected_tables

        with open(f'./data/{dataset}/{section}_new.json', 'w') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    new_map = {}
    for k, v in entities_map.items():
        new_map[k.lower()] = v.lower()

    with open(f'./data/{dataset}/entities_map.json', 'w') as f:
        json.dump(new_map, f, ensure_ascii=False, indent=2)


if __name__ == '__main__':
    select_tables_and_process_schema('CSgSQL')
