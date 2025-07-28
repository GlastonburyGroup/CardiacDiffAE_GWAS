import matplotlib.pyplot as plt
from wordcloud import WordCloud
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer
import nltk

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

def gen_wordcloud(df, filename, df_is_indtraits=False, col='GWAS Trait', background_colour="white", max_words=5000, contour_width=3, contour_colour='steelblue', width=800, height=400):
    traits = df[col]

    # Text preprocessing
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()

    def preprocess_text(text):
        tokens = word_tokenize(text.lower())
        new_tokens = []
        for token in tokens:
            if token.isalpha():
                lemmatized_token = lemmatizer.lemmatize(token)
                if lemmatized_token not in stop_words:
                    new_tokens.append(lemmatized_token)
            else:
                new_tokens.append(token)
        return new_tokens

    processed_traits = traits.apply(preprocess_text)    
    long_string = ' '.join(['_'.join(doc) if df_is_indtraits else ' '.join(doc) for doc in processed_traits])
    
    if bool(long_string):
        wordcloud = WordCloud(background_color=background_colour, max_words=max_words, contour_width=contour_width, contour_color=contour_colour, width=width, height=height)
        wordcloud.generate(long_string)
        wordcloud.to_file(filename)

        return long_string, wordcloud
    else:
        return None, None