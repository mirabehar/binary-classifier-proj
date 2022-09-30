import json
import gzip
import csv
import pandas as pd

'''
This module contains methods to pre-process the dataset
and save the new cleaned balanced dataset to a csv file in /data
'''
## GLOBAL VARS

## WAN datasets: book, genre, review
review_ds = 'data/goodreads_reviews_spoiler.json.gz'
book_info_ds = 'data/goodreads_books.json.gz'
genre_ds = 'data/goodreads_book_genres_initial.json'

## number of spoiler sentences in review_ds
_TOT_NUM_SPOIL = 569724
## by genre
TOT_NUM_SPOIL_ROM = 87917
TOT_NUM_SPOIL_FAN = 222828
TOT_NUM_SPOIL_YA = 11259

## Select which to use
TOT_NUM_SPOIL = _TOT_NUM_SPOIL

## INIT DICTS

book_to_genre = {}
book_to_author = {}
book_to_date_pub = {}


## MAKE DICTS

for line in open(genre_ds,"r"):
        jent = json.loads(line)
        bookID = jent["book_id"]

        ## get most popular genre label out of list of possible genres
        top_genre = jent["genres"]
        if top_genre == {}:
            continue

        ## genres have multiple names, pick first as ID
        ## eg. mystery-crime
        genre = sorted(top_genre, key=top_genre.get, reverse=True)[0].split(",")[0]

        # ##limit to these most popular genres:
        # if genre not in ["fantasy"]:
        #     continue

        book_to_genre[bookID] = genre



with gzip.open(book_info_ds) as f:
        for jentry in f:
            book_info = json.loads(jentry)
            bookID = book_info['book_id']
            ## get first auth if exists
            if len(book_info['authors']) == 0:
                continue

            auth = book_info['authors'][0]['author_id']
            date_pub = book_info['publication_year']

            book_to_author[bookID] = auth
            book_to_date_pub[bookID] = date_pub




def make_balanced_csv(dataset, csv_file, num_revs=None):
    '''
    Create csv from input json.gz dataset
    '''
    entry_count = 0
    num_spoil = 0
    num_no_spoil = 0

    csvf = open(csv_file, 'w', encoding='UTF8')
    header = ['reviewID', 'userID', 'bookID', 'rating', 
            'date_published','authorID', 'genre',
            'sent', 's_loc', 's_spoiler']
    
    writer = csv.writer(csvf)
    writer.writerow(header)

    with gzip.open(dataset) as f:
        for jentry in f:
            jrev = json.loads(jentry)

            ## Skip entries with no spoilers
            if jrev['has_spoiler'] == 'False':
                continue

            entry_count += 1
            reviewID = jrev['review_id']
            userID = jrev['user_id']
            bookID = jrev['book_id']
            stars = jrev['rating']


            if (bookID not in book_to_author.keys()
                or bookID not in book_to_date_pub.keys()
                or bookID not in book_to_genre.keys()):
                ## no entry for bookID, skip
                continue

            # genre = '##'
            # date_pub = '##'
            # auth = '##'
            genre = book_to_genre[bookID]
            date_pub = book_to_date_pub[bookID]
            auth = book_to_author[bookID]


            rev_info = [reviewID, userID, bookID, stars, date_pub, auth, genre]

            ## per sentence data
            for i, sent in enumerate(jrev['review_sentences']):
                sent_info=[]

                if sent[0] == 1:
                    num_spoil += 1
                elif sent[0] == 0:
                    num_no_spoil += 1

                ## reached 50-50 split: stop gathering no-spoil sents
                if num_no_spoil > TOT_NUM_SPOIL and sent[0] == 0:
                    continue

                sent_info = [sent[1], i, sent[0]]
                writer.writerow(rev_info + sent_info)

            
            # FOR DEV ONLY - break if reaches the nth entry
            if (num_revs is not None) and (entry_count > num_revs):
               break

    print("Num entries: " , entry_count)





if __name__=="__main__":
    csv_file = 'data/balanced_revs.csv'
    # make_balanced_csv(review_ds, csv_file)

    ## spoiler ratio
    df = pd.read_csv(csv_file)
    count0 = df[(df.s_spoiler == 0)].count()[0]
    count1 = df[(df.s_spoiler == 1)].count()[0]
    print(csv_file, "count0: ", count0, "\n",
                    "count1: ", count1, "\n",
                    "ratio: ", count1/(count0+count1))
    
    ## STATISTICS
    # df = pd.read_csv('data/balanced_revs.csv')
    
    # print(df["genre"].value_counts())

    # genres = ["fantasy",
    #         "romance",            
    #         "young-adult",
    #         "fiction",
    #         "mystery",
    #         "comics",
    #         "history",
    #         "non-fiction",
    #         "children",
    #         "poetry"]


    # for g in genres:
    #     print("0 count: ", g, df[(df.genre == g) & (df.s_spoiler == 0)].count())
    #     print("1 count: ", g, df[(df.genre == g) & (df.s_spoiler == 1)].count())