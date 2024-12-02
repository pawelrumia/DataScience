import pandas as pd
from collections import Counter

imdb = pd.read_csv("imdb_preprocessed.csv")

positive_reviews = imdb[imdb['sentiment'] == 'positive']['review_processed']
negative_reviews = imdb[imdb['sentiment'] == 'negative']['review_processed']

all_words_pos = ' '.join(positive_reviews).split()
all_words_neg = ' '.join(negative_reviews).split()

word_freq_pos = Counter(all_words_pos)
word_freq_neg = Counter(all_words_neg)

positive_common_words = word_freq_pos.most_common(20)
negative_common_words = word_freq_neg.most_common(20)

print("20 najczęstszych słów w recenzjach pozytywnych:")
print(positive_common_words)

print("\n20 najczęstszych słów w recenzjach negatywnych:")
print(negative_common_words)