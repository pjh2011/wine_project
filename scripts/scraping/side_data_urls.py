import pandas as pd


def read_files(names):

    for name in names:
        tmp = pd.read_csv(
            '../../data/dataframes/' + name)

        if 'df' not in vars():
            df = tmp
        else:
            df = pd.concat([df, tmp])

    return df


def jsonify_n(start, finish, np_array):
    s = '  "urls":['

    for i in range(start, finish + 1):
        if i > start:
            s += ','

        s += '"' + 'http://www.cellartracker.com/wine.asp?iWine=' + \
            str(np_array[i]) + '"'

    s += ']'

    print "{"
    # print '
    # "urls":["http://www.cellartracker.com/notes.asp?iWine=947","http://www.cellartracker.com/notes.asp?iWine=948"]'
    print s
    print "}"


if __name__ == "__main__":
    reviews = read_files(['first_1000.csv', 'next_9000.csv'])
    reviews = reviews[~reviews['review_text'].isnull()]

    ids = reviews.id.unique()

    start = 7001
    finish = len(ids) - 1
    jsonify_n(start, finish, reviews['id'].unique())

    # http://www.cellartracker.com/wine.asp?iWine=874157
