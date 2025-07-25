import re
import string
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.tokenize import word_tokenize


def preprocess_texts(
        query: str,
        lang: str = 'english',
        remove_punctuation: bool = True,
        remove_stopwords: bool = True,
        use_lemmatization: bool = False,
        use_stemming: bool = False,
) -> str:
    """
    Preprocess a user query for RAG retrieval.

    Args:
        query: Input query string.
        lang: Language for stopwords ('english', 'french', etc.).
        remove_punctuation: Whether to remove punctuation.
        remove_stopwords: Whether to remove stopwords.
        use_lemmatization: Use WordNet lemmatization (slower but more accurate).
        use_stemming: Use Porter stemming (faster but aggressive).

    Returns:
        Preprocessed query string.
    """
    # Lowercase and strip whitespace
    query = query.strip().lower()

    if lang == 'english':
        # Remove punctuation (optional)
        if remove_punctuation:
            query = re.sub(f'[{re.escape(string.punctuation)}]', ' ', query)

        # Tokenize
        tokens = word_tokenize(query)

        # Remove stopwords (optional)
        if remove_stopwords:
            stop_words = set(stopwords.words(lang))
            tokens = [word for word in tokens if word not in stop_words]

        # Apply stemming or lemmatization (optional)
        if use_stemming:
            stemmer = PorterStemmer()
            tokens = [stemmer.stem(word) for word in tokens]
        elif use_lemmatization:
            lemmatizer = WordNetLemmatizer()
            tokens = [lemmatizer.lemmatize(word) for word in tokens]

        # Rejoin into a single string
        return ' '.join(tokens)
    else:
        return query
