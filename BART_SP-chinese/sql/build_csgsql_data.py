import os

import json
from tqdm import tqdm
import sentencepiece as spm


def read_database_schema(database_schema_filename, schema_tokens, column_names, database_schemas_dict):
    with open(database_schema_filename) as f:
        database_schemas = json.load(f)

    def get_schema_tokens(table_schema):
        column_names_surface_form = []
        column_names = []
        column_names_original = table_schema['column_names_original']
        table_names = table_schema['table_names']
        table_names_original = table_schema['table_names_original']
        for i, (table_id, column_name) in enumerate(column_names_original):
            if table_id >= 0:
                table_name = table_names_original[table_id]
                column_name_surface_form = '{}.{}'.format(table_name, column_name)
            else:
                # this is just *
                column_name_surface_form = column_name
            column_names_surface_form.append(column_name_surface_form.lower())
            column_names.append(column_name.lower())

        # also add table_name.*
        for table_name in table_names_original:
            column_names_surface_form.append('{}.*'.format(table_name.lower()))

        return column_names_surface_form, column_names

    for table_schema in database_schemas:
        database_id = table_schema['db_id']
        database_schemas_dict[database_id] = table_schema
        schema_tokens[database_id], column_names[database_id] = get_schema_tokens(table_schema)

    return schema_tokens, column_names, database_schemas_dict


def extract_input_and_output(example_lines):
    inputs = []
    outputs = []

    database_schema_filename = './data/CSgSQL/db_schema_new.json'

    schema_tokens = {}
    column_names = {}
    database_schemas = {}
    schema_tokens, column_names, database_schemas = read_database_schema(database_schema_filename, schema_tokens,
                                                                         column_names, database_schemas)
    max_len=-1
    for line in tqdm(example_lines):
        item = json.loads(line.strip())
        question = item['utterance']
        schema = database_schemas[item['database_id']]

        column_names = []
        selected_tables = []
        for t, c in zip(schema['column_types'], schema['column_names_original']):
            if c[0] == -1:
                column_names.append("{0} {1}".format(t, c[1].lower()))
            elif c[0] in item['selected_tables']:
                first = False
                if c[0] not in selected_tables:
                    first = True
                    selected_tables.append(c[0])
                    tab = schema['table_names_original'][c[0]].lower()
                    column_names.append("<T> {0}".format(tab.lower()))

                column_name = c[1].lower()
                head = '<C> ' if first else ''
                column_names.append("{0}{1} {2}".format(head, t, column_name))

        column_names = ' | '.join(column_names)

        source_sequence=f"<C> {column_names} | <Q> {question.lower()}"
        target_sequence = item['sql'].lower()

        outputs.append(target_sequence)
        inputs.append(source_sequence)
        max_len = max(max_len, len(source_sequence.split()))

    assert len(inputs) == len(outputs)
    print(max_len, len(inputs), len(outputs))
    return inputs, outputs


def read_dataflow_dataset(file_path, out_folder, mode, spm_model_path):

    train_out_path = os.path.join(out_folder, mode)
    train_src_writer = open(train_out_path + ".src", "w", encoding="utf8")
    train_tgt_writer = open(train_out_path + ".tgt", "w", encoding="utf8")

    with open(file_path, "r") as data_file:
        lines = data_file.readlines()
        data_input, data_output = extract_input_and_output(lines)
        train_src_writer.write("\n".join(data_input))
        train_tgt_writer.write("\n".join(data_output))
    train_src_spm_writer = open(train_out_path + ".spm.src", "w", encoding="utf8")
    train_tgt_spm_writer = open(train_out_path + ".spm.tgt", "w", encoding="utf8")

    sp = spm.SentencePieceProcessor()
    sp.Load(model_file=spm_model_path)
    for src, tgt in zip(data_input, data_output):
        train_src_spm_writer.write(' '.join(sp.encode(src, out_type=str)) + '\n')
        train_tgt_spm_writer.write(' '.join(sp.encode(tgt, out_type=str)) + '\n')


def build_csgsql_data(overnight_path, out_path, spm_model_path):
    for mode in ["train", "dev"]:
        file_path = os.path.join(overnight_path, "{}.json".format(mode))
        out_folder = out_path
        if not os.path.exists(out_folder):
            os.makedirs(out_folder)
        read_dataflow_dataset(file_path, out_folder, mode, spm_model_path)


if __name__ == '__main__':
    OVERNIGHT_PATH = f"./data/CSgSQL_sl"
    OUT_PATH = f"./dataset_post/csgsql"
    SPM_MODEL_PATH = './data/ptm/sentence.bpe.model'
    build_csgsql_data(OVERNIGHT_PATH, OUT_PATH, SPM_MODEL_PATH)

