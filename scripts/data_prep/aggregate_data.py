import pandas as pd
import os
from unidecode import unidecode, unidecode_expect_nonascii


def merge_csvs():
    cols = ['results_wine_name', 'results_wine_name_url',
            'results_wine_reviews_name']

    for i in range(10):
        tmp = pd.read_csv(
            '../../data/raw/first_1000/run_results' + str(i) + '.csv')
        tmp = tmp[cols]

        if 'df' not in vars():
            df = tmp
        else:
            df = pd.concat([df, tmp])

    df.columns = ['wine_name', 'id', 'review_text']

    not_null = ~df['review_text'].isnull()

    df = df[not_null]

    df.drop_duplicates(inplace=True)

    df['wine_name'] = df['wine_name'].apply(
        lambda x: unidecode_expect_nonascii(x))
    df['review_text'] = df['review_text'].apply(
        lambda x: unidecode_expect_nonascii(x))
    df['id'] = df['id'].apply(lambda x: int(x.split('?iWine=')[1]))

    return df


def merge_jsons(folder_name):
    path = '../../data/raw/' + folder_name

    for fname in os.listdir(path):
        full_path = path + '/' + fname
        tmp = pd.read_json(full_path)
        if 'df' not in vars():
            df = tmp
        else:
            df = pd.concat([df, tmp])

    num_els = df['results'].apply(lambda x: len(x))

    df = df[num_els == 3]

    names_list = []
    ids_list = []
    reviews_list = []

    for i in range(df.shape[0]):
        entry = df['results'].iloc[i]

        name = unidecode_expect_nonascii(entry['wine_name'])
        wine_id = int(entry['wine_name_url'].split('?iWine=')[1])
        reviews = entry['wine_reviews']

        n_reviews = len(reviews)

        names_list += [name] * n_reviews
        ids_list += [wine_id] * n_reviews

        for j in range(n_reviews):
            reviews_list.append(unidecode_expect_nonascii(reviews[j]['name']))

    unpacked = pd.DataFrame(zip(names_list, ids_list, reviews_list))
    unpacked.columns = ['wine_name', 'id', 'review_text']

    not_null = ~unpacked['review_text'].isnull()

    unpacked = unpacked[not_null]

    unpacked.drop_duplicates(inplace=True)

    return unpacked

if __name__ == "__main__":
    # df = merge_csvs()
    # df.to_csv('../../data/dataframes/first_1000.csv', index=False)

    df = merge_jsons('next_9000')
    df.to_csv('../../data/dataframes/next_9000.csv', index=False)
