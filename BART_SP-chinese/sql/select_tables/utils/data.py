from __future__ import annotations
from typing import Dict, List, Any
from torch.utils.data import DataLoader
import jieba
import json


class Question(object):
    def __init__(self, question: str, tables: List[int], question_id: str, schema: List[str] = None):
        self.question = question
        self.tables = tables
        self.question_id = question_id
        # [table_name [SEP] column_name1 | column_type1 | column_value1(or [UNK]) [SEP] ...]
        self.schema = schema

    def __hash__(self) -> int:
        return hash(self.question)


class DataManager(object):
    def __init__(self):
        self.schema: Dict[int, str] = {}
        self.questions: List[Question] = []
        self.foreign_key: Dict[int, List[int]] = {}
        self.column_belong: Dict[int, int] = {}
        self.table_column: Dict[int, List[int]] = {}
        self.table_name: Dict[int, str] = {}
        self.column_name: Dict[int, str] = {}

        self.type_dict: Dict[str, str] = {
            'text': '文本', 'time': '日期', 'number': '数值'}
        self.table_name_dict: Dict[str, str] = {}
        self.column_name_dict: Dict[str, str] = {}
        self.table_content: Dict[str, Dict[str, List[Any]]] = {}
        self.column_type_dict: Dict[str, str] = {}

    @staticmethod
    def load(problem_path: str, schema_path: str, content_path: str, max_length) -> DataManager:
        dm = DataManager()

        with open(schema_path, 'r', encoding='utf-8') as f:
            data = json.load(f)[0]

            table_name = data['table_names']
            table_name_original = data['table_names_original']
            for i in range(len(table_name)):
                dm.schema[i] = table_name[i]
                dm.table_name[i] = table_name[i]
                dm.table_name_dict[table_name_original[i]] = table_name[i]
                dm.foreign_key[i] = []

            column_idx = data['column_names']
            column_name_original = data['column_names_original']
            column_type = data['column_types']
            for i in range(len(column_idx)):
                d = column_idx[i]
                if d[0] == -1:
                    continue

                if not(d[0] in dm.table_column):
                    dm.table_column[d[0]] = []
                dm.table_column[d[0]].append(i)
                dm.column_name[i] = d[1]
                dm.column_belong[i] = d[0]
                # type passes -1
                dm.column_name_dict[column_name_original[i][1]
                                    ] = d[1]
                dm.column_type_dict[d[1]] = dm.type_dict[column_type[i - 1]]

            foreign_key = data['foreign_keys']
            for k in foreign_key:
                table1 = dm.column_belong[k[0]]
                table2 = dm.column_belong[k[1]]
                if not(table2 in dm.foreign_key[table1]):
                    dm.foreign_key[table1].append(table2)
                if not(table1 in dm.foreign_key[table2]):
                    dm.foreign_key[table2].append(table1)

        with open(content_path, 'r', encoding='utf-8') as f:
            data = json.load(f)[0]['tables']
            for table_name in data:
                current_table_name = dm.table_name_dict[table_name]
                current_table = data[table_name]
                current_header = current_table['header']
                dm.table_content[current_table_name] = {
                    dm.column_name_dict[header]: [] for header in current_header}

                for cell in current_table['cell']:
                    for i in range(len(cell)):
                        dm.table_content[current_table_name
                                         ][dm.column_name_dict[current_header[i]]].append(cell[i])

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

        with open(problem_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            cnt = 0
            for d in data:
                current_question = d['question']
                current_table_idx = []
                if 'sql' in d:
                    find_all_tabs(d['sql'], current_table_idx)

                dm.questions.append(
                    Question(current_question, current_table_idx, d['question_id']))

                if max_length is not None:
                    cnt += 1
                    if cnt >= max_length:
                        break

        for i in range(len(dm.questions)):
            question: List[str] = jieba.lcut(dm.questions[i].question)
            current_schema_list: List[str] = []
            for j in range(len(dm.table_name)):
                current_table_name = dm.table_name[j]
                current_schema = current_table_name + '[SEP]'
                current_content = dm.table_content[current_table_name]

                for column_name in current_content:
                    flag = False

                    for content in current_content[column_name]:
                        if _check_string_contain(question, str(content)):
                            current_schema += ' ' + column_name + ' ' + \
                                dm.column_type_dict[column_name] + \
                                ' ' + str(content)
                            flag = True
                            break
                    if flag:
                        continue

                    if _check_string_contain(question, column_name):
                        current_schema += ' ' + column_name + ' ' + \
                            dm.column_type_dict[column_name]

                current_schema_list.append(current_schema)
            dm.questions[i].schema = current_schema_list

        return dm

    def get_schema(self) -> List[str]:
        return [self.schema[i] for i in range(len(self.schema))]

    def package(self, batch_size: int, shuffle: bool = True):
        questions = []
        tables = []
        for q in self.questions:
            questions.append(q.question)
            tables.append(q.tables)

        return DataLoader(dataset=_DataSet(questions, tables),
                          batch_size=batch_size,
                          shuffle=shuffle,
                          collate_fn=_collate_fn)

    def __len__(self):
        return len(self.questions)


class _DataSet(object):
    def __init__(self, questions: List[str], tables: List[List[int]]):
        self.questions = questions
        self.tables = tables

    def __len__(self):
        return len(self.questions)

    def __getitem__(self, idx: int):
        return self.questions[idx], self.tables[idx]


def _collate_fn(batch):
    attr_count = len(batch[0])
    ret = [[] for i in range(attr_count)]

    for i in range(len(batch)):
        for j in range(attr_count):
            ret[j].append(batch[i][j])

    return ret


def _check_string_contain(s: List[str], t: str) -> bool:
    if t.isdigit():
        return False
    for x in s:
        if x in t:
            return True
    return False
