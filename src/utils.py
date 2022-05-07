import os
import re
import pandas as pd
import numpy as np


def text_preproc(s):
    if not pd.isna(s):
        s = s.lower()
        s = re.sub(r'\s([?.!,"](?:\s|$))', r'\1', s)
        return s
    else:
        return r' '


def csv_reader(path):
    df = pd.read_csv(path, sep='\t', on_bad_lines='skip')
    return df


def csv_save(df, path):
    df.to_csv(path, index=False, encoding='utf-8', sep='\t')


def prepare_data(dir_path, filename):
    filepath = os.path.join(dir_path, filename)
    data = csv_reader(filepath)

    data.context_0 = data.context_0.apply(text_preproc)
    data.reply = data.reply.apply(text_preproc)
    data.label = data.label.apply(text_preproc)

    dataset = pd.concat([data.context_0, data.reply, data.label],
                        axis=1, keys=['context_0', 'reply', 'label'])

    # delete bad lines
    dataset = dataset.drop(labels=[671, 754, 1258, 1259, 1568, 2535, 4614, 5015, 12761, 14316, 16884,
                                   17003, 19445, 22532, 26915, 33753, 33754, 33755, 34507, 35663, 36832,
                                   38300, 38511, 41324, 43815, 44856, 47081, 51221, 51835, 52423, 54317], axis=0)

    questions = dataset['context_0'].unique()
    print(f'len questions: {len(questions)}')
    desired_number = int(len(questions) * 0.33)
    print(f'desired_number: {desired_number}')
    test_questions = np.random.choice(questions, size=desired_number, replace=False)
    print(f'len test_questions: {len(test_questions)}')
    test_dataset = dataset.loc[dataset['context_0'].isin(test_questions)]
    print(f'len unique test_questions in pandas: {len(test_dataset.context_0.unique())}')

    with open(os.path.join(dir_path, f'{filename}_test_questions.txt'), 'w') as f:
        for question in test_questions:
            f.write(question + '\n')
    csv_save(test_dataset, os.path.join(dir_path, f'{filename}_test_dataset.tsv'))
    csv_save(dataset, os.path.join(dir_path, f'{filename}_clean.tsv'))


if __name__ == '__main__':

    filepath = f'/Users/d.volf/Documents/Projects/Bot/data'
    source_filename = f'good_clean.tsv'

    prepare_data(filepath, source_filename)
    # test_filename = 'good_test_dataset.tsv'
    # test_filename = 'good_test_dataset1.csv'
