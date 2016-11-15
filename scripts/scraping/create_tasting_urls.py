import pandas as pd


def read_urls():
    countries = ['argentina', 'australia', 'chile', 'france', 'germany',
                 'italy', 'portugal', 'spain', 'usa', 'zaf']

    cols = ['results_wine_hovercard_url',
            'results_wine_hovercard_wine_name',
            'results_wine_hovercard_num_bottles',
            'results_wine_hovercard_score']

    for c in countries:
        tmp = pd.read_csv('url_results/run_results_' + c + '.csv')
        tmp = tmp[cols]
        tmp['country'] = c

        if 'df' not in vars():
            df = tmp
        else:
            df = pd.concat([df, tmp])

    df.columns = ['url', 'name', 'n_bottles', 'score', 'country']

    df['n_bottles'] = df['n_bottles'].apply(
        lambda x: int(x.split()[0].replace(',', '')))

    df['id'] = df['url'].apply(lambda x: x.split('iWine=')[1])
    df.sort_values(by='n_bottles', ascending=False, inplace=True)

    return df


def jsonify_n(start, finish, df):
    s = '  "urls":['

    for i in range(start, finish + 1):
        if i > start:
            s += ','

        s += '"' + 'http://www.cellartracker.com/notes.asp?iWine=' + \
            df['id'].iloc[i] + '"'

    s += ']'

    print "{"
    # print '
    # "urls":["http://www.cellartracker.com/notes.asp?iWine=947","http://www.cellartracker.com/notes.asp?iWine=948"]'
    print s
    print "}"


df = read_urls()

# have ten jobs running 1-100, 101-200, ..., 901-1000
# have 9 more jobs 1001-2000, ..., 9001-10000
start = 9001
finish = 10000

jsonify_n(start, finish, df)
