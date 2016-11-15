import pandas as pd
import os


def merge_csvs():
    cols = ['results_wine_name', 'results_wine_reviews_name']

    for i in range(10):
        tmp = pd.read_csv(
            '../../data/raw/first_1000/run_results' + str(i) + '.csv')
        tmp = tmp[cols]

        if 'df' not in vars():
            df = tmp
        else:
            df = pd.concat([df, tmp])

    df.columns = ['wine_name', 'review_text']

    not_null = ~df.isnull().iloc[:, 1]

    df = df[not_null]

    df.drop_duplicates(inplace=True)

    return df


def merge_jsons(folder_name):
    path = '../../data/raw/' + folder_name
    return os.listdir(path)

if __name__ == "__main__":
    # df = merge_csvs()

    # df.to_csv('../data/dataframes/first_1000.csv', index=False)
    print merge_jsons('next_9000')
