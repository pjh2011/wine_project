import pandas as pd
import os
from unidecode import unidecode, unidecode_expect_nonascii


def merge_csvs():
    # columns to keep from scraped data
    cols = ['results_wine_name', 'results_wine_name_url',
            'results_wine_reviews_name']

    # ten files initially scraped, iterate over them and merge
    for i in range(10):
        tmp = pd.read_csv(
            '../../data/raw/first_1000/run_results' + str(i) + '.csv')
        tmp = tmp[cols]

        if 'df' not in vars():
            df = tmp
        else:
            df = pd.concat([df, tmp])

    # rename columns
    df.columns = ['wine_name', 'id', 'review_text']

    # remove null values
    not_null = ~df['review_text'].isnull()

    df = df[not_null]

    # drop duplicates (if created by scraper)
    df.drop_duplicates(inplace=True)

    # decode unicode characters
    df['wine_name'] = df['wine_name'].apply(
        lambda x: unidecode_expect_nonascii(x))
    df['review_text'] = df['review_text'].apply(
        lambda x: unidecode_expect_nonascii(x))

    # split the wine id from the url
    df['id'] = df['id'].apply(lambda x: int(x.split('?iWine=')[1]))

    return df


def merge_jsons(folder_name):
    path = '../../data/raw/' + folder_name

    # loop over files in the folder, merge the dataframes
    for fname in os.listdir(path):
        full_path = path + '/' + fname
        tmp = pd.read_json(full_path)
        if 'df' not in vars():
            df = tmp
        else:
            df = pd.concat([df, tmp])

    # each entry is a dictionary, only length 3 was populated with data from
    # the scraper
    num_els = df['results'].apply(lambda x: len(x))

    df = df[num_els == 3]

    # create empty lists, to be appended to by expanding each dictionary
    names_list = []
    ids_list = []
    reviews_list = []

    # loop over each row, open the dictionary to get all reviews associated
    # with each wine...then create lists of the reviews, corresponding names,
    # and ids. then create df from the lists
    for i in range(df.shape[0]):
        entry = df['results'].iloc[i]

        # decode unicode chars
        name = unidecode_expect_nonascii(entry['wine_name'])

        # split wine id from url
        wine_id = int(entry['wine_name_url'].split('?iWine=')[1])

        # get dictionary of wine reviews for each wine
        reviews = entry['wine_reviews']

        n_reviews = len(reviews)

        # append the names and ids to each list n_reviews times
        # when we create dataframe these will then be in proper rows associated
        # with each review
        names_list += [name] * n_reviews
        ids_list += [wine_id] * n_reviews

        # loop over the reviews, decode the unicode chars, then append to list
        for j in range(n_reviews):
            reviews_list.append(unidecode_expect_nonascii(reviews[j]['name']))

    # create dataframe, rename columns
    unpacked = pd.DataFrame(zip(names_list, ids_list, reviews_list))
    unpacked.columns = ['wine_name', 'id', 'review_text']

    # remove rows with no review text
    not_null = ~unpacked['review_text'].isnull()

    unpacked = unpacked[not_null]

    # drop duplicates
    unpacked.drop_duplicates(inplace=True)

    return unpacked

if __name__ == "__main__":
    # df = merge_csvs()
    # df.to_csv('../../data/dataframes/first_1000.csv', index=False)

    df = merge_jsons('next_9000')
    df.to_csv('../../data/dataframes/next_9000.csv', index=False)
