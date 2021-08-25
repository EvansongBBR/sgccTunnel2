from sql.select_tables.utils.data import DataManager
from nltk.util import ngrams
from typing import List
from tqdm import tqdm
from json import dump
from sql.select_tables.config import parse_args


def _ngram_process(s: str, n: List[int]) -> List[str]:
    ngram = []
    for x in n:
        ngram = ngram + list(ngrams(s, x))
    ngram = ["".join(x) for x in ngram]
    ngram = list(set(ngram))
    return ngram


def _check_same_gram(a: List[str], b: List[str]) -> int:
    a.sort()
    b.sort()

    i = 0
    j = 0
    cnt = 0
    while i < len(a) and j < len(b):
        if a[i] < b[j]:
            i += 1
        elif a[i] > b[j]:
            j += 1
        else:
            cnt += 1
            i += 1
            j += 1
    return cnt


def get_candidate_tables(predict_data_path, schema_path, content_path,
                         ngram=(1, 2, 3, 4),
                         answer_size=6,
                         return_eval=False,
                         max_length=None):
    dm: DataManager = DataManager.load(
        predict_data_path, schema_path, content_path, max_length)
    lam = [0.1, 0.1, 0.1, 5, 5, 5, 5, 5, 5, 5]
    used_rec_gram = [True, True, True, False, False, False, False, False, False, False]

    real_answer: List[List[int]] = []
    predict_answer: List[List[int]] = []
    for q in tqdm(dm.questions):
        current_question = q.question
        current_schema = q.schema
        current_real_answer = q.tables

        real_answer.append(current_real_answer)
        predict_answer.append([])
        candidate_tables = list(range(len(current_schema)))
        question_ngram = _ngram_process(current_question, ngram)
        rec_gram = []

        for i in range(answer_size):
            best_answer = -1
            best_score = -1
            best_gram = []
            for t in candidate_tables:
                if used_rec_gram[i]:
                    current_question_gram = list(
                        set(question_ngram) - set(rec_gram))
                else:
                    current_question_gram = question_ngram

                current_table = current_schema[t]
                table_name_ngram = _ngram_process(
                    current_table.split('[SEP]')[0], ngram)
                schema_ngram = _ngram_process(current_table, ngram)
                current_score = lam[i] * _check_same_gram(
                    current_question_gram, schema_ngram) + _check_same_gram(current_question_gram, table_name_ngram)
                if current_score > best_score:
                    best_score = current_score
                    best_answer = t
                    best_gram = table_name_ngram

            if i == 0:
                candidate_tables = []
            predict_answer[-1].append(best_answer)
            candidate_tables = candidate_tables + dm.foreign_key[best_answer]
            candidate_tables = list(
                set(candidate_tables) - set(predict_answer[-1]))
            rec_gram = list(set(rec_gram + best_gram))

    recall = 0
    example_recall = 0
    # f1 = 0
    result = []
    for i in range(len(predict_answer)):
        cnt = _check_same_gram(predict_answer[i], real_answer[i])
        current_recall = cnt / len(real_answer[i])
        example_recall += 1 if current_recall == 1 else 0
        recall += current_recall

        predict_table = [dm.table_name[x]
                         for x in predict_answer[i]]
        real_table = [dm.table_name[x]
                      for x in real_answer[i]]
        question = dm.questions[i].question

        # if cnt != len(real_answer[i]):
        result.append(
            {'question': question,
             'question_id': dm.questions[i].question_id,
             'real_tables': real_table,
             'predict_tables': predict_table,
             'predict_tables_id': predict_answer[i]})

    if return_eval:
        return result, {'recall (table level)': recall/len(predict_answer),
                        'recall (example level)': example_recall/len(predict_answer)}
    else:
        return result


def main():
    args = parse_args()
    result, eval = \
        get_candidate_tables(args.predict_data_path,
                             args.schema_path,
                             args.content_path,
                             args.ngram,
                             args.answer_size,
                             return_eval=True,
                             max_length=args.max_length)
    with open(args.result_path, 'w', encoding='utf-8') as f:
        dump(result, f, ensure_ascii=False, indent=4)

    print('\n'.join([f'{k}: {v}' for k,v in eval.items()]))


if __name__ == '__main__':
    main()
