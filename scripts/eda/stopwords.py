
# words to keep
# words inappropriately removed by just taking out words from names
wine_words = ['chalk', 'cherry', 'chocolate', 'crema', 'dark', 'dry', 'flor',
              'flora', 'flowers', 'forest', 'graham', 'grapes',
              'iron', 'mineral', 'moss', 'nickel', 'oak', 'oaks', 'profile',
              'sea', 'smoke', 'stone', 'stones', 'strong', 'sugar', 'turkey',
              'unoaked', 'velvet', 'wood']


# generated looking at top words within clusters
wine_names_places = set(['yquem', 'bordeaux', 'crus', 'cab', 'ugc', 'cali',
                         'california', 'ny', ' hotel', 'st julien', 'union',
                         'drake', 'palace', 'chicago', 'illinois',
                         'convention', 'centre', 'primeur', 'paulee',
                         'new york', 'york', 'francisco', 'london',
                         'metropolitan', 'pavilion', 'lynch', 'bages',
                         'new world', 'burgundy', 'vega', 'sicilia',
                         'julien', 'zin', 'bottle', 'bottles', 'cdp',
                         'chateauneuf', 'bouteille', 'kirsch', 'wa', 'usa'
                         'canada', 'og', 'av', 'och', 'kirsch'])

# source: http://www.ranks.nl/stopwords/french
# ran strings through unidecode to convert to ascii
french_stop_words = set(['alors', 'au', 'aucuns', 'aussi', 'autre', 'avant',
                         'avec', 'avoir', 'bon', 'car', 'ce', 'cela', 'ces',
                         'ceux', 'chaque', 'ci', 'comme', 'comment', 'dans',
                         'des', 'du', 'dedans', 'dehors', 'depuis', 'devrait',
                         'doit', 'donc', 'dos', 'dA(c)but', 'elle', 'elles',
                         'en', 'encore', 'essai', 'est', 'et', 'eu', 'fait',
                         'faites', 'fois', 'font', 'hors', 'ici', 'il', 'ils',
                         'je', 'juste', 'la', 'le', 'les', 'leur', 'lA', 'ma',
                         'maintenant', 'mais', 'mes', 'mine', 'moins', 'mon',
                         'mot', 'mAame', 'ni', 'nommA(c)s', 'notre', 'nous',
                         'ou', 'oA1', 'par', 'parce', 'pas', 'peut', 'peu',
                         'plupart', 'pour', 'pourquoi', 'quand', 'que', 'quel',
                         'quelle', 'quelles', 'quels', 'qui', 'sa', 'sans',
                         'ses', 'seulement', 'si', 'sien', 'son', 'sont',
                         'sous', 'soyez', 'sujet', 'sur', 'ta', 'tandis',
                         'tellement', 'tels', 'tes', 'ton', 'tous', 'tout',
                         'trop', 'trA"s', 'tu', 'voient', 'vont', 'votre',
                         'vous', 'vu', 'ASSa', 'A(c)taient', 'A(c)tat',
                         'A(c)tions', 'A(c)tA(c)', 'Aatre'])

# souce:
# https://github.com/scikit-learn/scikit-learn/blob/master/sklearn/...
# ...feature_extraction/stop_words.py

english_stop_words = set([
    "a", "about", "above", "across", "after", "afterwards", "again", "against",
    "all", "almost", "alone", "along", "already", "also", "although", "always",
    "am", "among", "amongst", "amoungst", "amount", "an", "and", "another",
    "any", "anyhow", "anyone", "anything", "anyway", "anywhere", "are",
    "around", "as", "at", "back", "be", "became", "because", "become",
    "becomes", "becoming", "been", "before", "beforehand", "behind", "being",
    "below", "beside", "besides", "between", "beyond", "bill", "both",
    "bottom", "but", "by", "call", "can", "cannot", "cant", "co", "con",
    "could", "couldnt", "cry", "de", "describe", "detail", "do", "done",
    "down", "due", "during", "each", "eg", "eight", "either", "eleven", "else",
    "elsewhere", "empty", "enough", "etc", "even", "ever", "every", "everyone",
    "everything", "everywhere", "except", "few", "fifteen", "fifty", "fill",
    "find", "fire", "first", "five", "for", "former", "formerly", "forty",
    "found", "four", "from", "front", "full", "further", "get", "give", "go",
    "had", "has", "hasnt", "have", "he", "hence", "her", "here", "hereafter",
    "hereby", "herein", "hereupon", "hers", "herself", "him", "himself", "his",
    "how", "however", "hundred", "i", "ie", "if", "in", "inc", "indeed",
    "interest", "into", "is", "it", "its", "itself", "keep", "last", "latter",
    "latterly", "least", "less", "ltd", "made", "many", "may", "me",
    "meanwhile", "might", "mill", "mine", "more", "moreover", "most", "mostly",
    "move", "much", "must", "my", "myself", "name", "namely", "neither",
    "never", "nevertheless", "next", "nine", "no", "nobody", "none", "noone",
    "nor", "not", "nothing", "now", "nowhere", "of", "off", "often", "on",
    "once", "one", "only", "onto", "or", "other", "others", "otherwise", "our",
    "ours", "ourselves", "out", "over", "own", "part", "per", "perhaps",
    "please", "put", "rather", "re", "same", "see", "seem", "seemed",
    "seeming", "seems", "serious", "several", "she", "should", "show", "side",
    "since", "sincere", "six", "sixty", "so", "some", "somehow", "someone",
    "something", "sometime", "sometimes", "somewhere", "still", "such",
    "system", "take", "ten", "than", "that", "the", "their", "them",
    "themselves", "then", "thence", "there", "thereafter", "thereby",
    "therefore", "therein", "thereupon", "these", "they", "thick", "thin",
    "third", "this", "those", "though", "three", "through", "throughout",
    "thru", "thus", "to", "together", "too", "top", "toward", "towards",
    "twelve", "twenty", "two", "un", "under", "until", "up", "upon", "us",
    "very", "via", "was", "we", "well", "were", "what", "whatever", "when",
    "whence", "whenever", "where", "whereafter", "whereas", "whereby",
    "wherein", "whereupon", "wherever", "whether", "which", "while", "whither",
    "who", "whoever", "whole", "whom", "whose", "why", "will", "with",
    "within", "without", "would", "yet", "you", "your", "yours", "yourself",
    "yourselves"])
