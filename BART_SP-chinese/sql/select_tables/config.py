import argparse
from typing import List

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--predict_data_path',
                        '-pdp',
                        type=str,
                        default='./data/CSgSQL/train.json',
                        help="path of test data")
    parser.add_argument('--schema_path', '-sp', type=str,
                        default='./data/CSgSQL/db_schema.json')
    parser.add_argument('--content_path', '-cp', type=str,
                        default='./data/CSgSQL/db_content.json', help='path of content')
    parser.add_argument('--result_path',
                        '-rp',
                        type=str,
                        default='./data/CSgSQL/select_table_result.json',
                        help="output path")
    parser.add_argument('--max_length', '-ml', type=int,
                        default=None, help='max lines of data to read')
    parser.add_argument('--ngram', '-n', type=List[int],
                        default=[1, 2, 3, 4], help='ngram parameter')
    parser.add_argument('--answer_size', '-as', type=int,
                        default=6, help="predict number of tables")

    args = parser.parse_args()
    return args
