import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# Ensure necessary NLTK datasets are downloaded
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('wordnet')

class TextCleaner:
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()
        
        # Add custom stopwords relevant to meeting transcripts
        self.custom_stopwords = {'um', 'uh', 'like', 'you know', 'so', 'yeah', 'okay', 'alright'}
        self.stop_words.update(self.custom_stopwords)

    def clean_text(self, text: str) -> str:
        """
        Performs basic NLP preprocessing pipeline (CO-4):
        1. Lowercasing
        2. Removing punctuation and special characters
        3. Tokenization
        4. Stopword removal
        5. Lemmatization
        """
        if not text:
            return ""

        # 1. Lowercasing
        text = text.lower()
        
        # Remove timestamps if present (e.g., [00:01:23] or 12:45:00)
        text = re.sub(r'\[?\d{2}:\d{2}(:\d{2})?\]?', '', text)
        
        # Remove speaker labels (e.g., Speaker 1:, John:)
        text = re.sub(r'\w+\s*\d*:\s', '', text)

        # 2. Removing punctuation
        text = text.translate(str.maketrans(''.join(string.punctuation), ' ' * len(string.punctuation)))
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()

        # 3. Tokenization
        try:
            tokens = word_tokenize(text)
        except LookupError:
            tokens = text.split()

        # 4 & 5. Stopword removal and Lemmatization
        cleaned_tokens = [
            self.lemmatizer.lemmatize(word)
            for word in tokens
            if word not in self.stop_words and len(word) > 1
        ]

        return ' '.join(cleaned_tokens)

    def extract_keywords(self, text: str, top_n: int = 10) -> list:
        """Extract most frequent meaningful words."""
        cleaned_text = self.clean_text(text)
        tokens = cleaned_text.split()
        
        freq_dist = nltk.FreqDist(tokens)
        return [word for word, freq in freq_dist.most_common(top_n)]

# Example usage for testing
if __name__ == "__main__":
    cleaner = TextCleaner()
    sample_transcript = """
    [00:01:23] John: Okay, so, let's get started. The main issue today is the database migration.
    [00:01:30] Sarah: Yeah, I agree. We need to finish it by next Friday, alright?
    [00:01:40] John: Um, definitely. I will assign the backend tasks to the engineering team.
    """
    
    print("Original Text:\n", sample_transcript)
    print("\nCleaned Text:\n", cleaner.clean_text(sample_transcript))
    print("\nTop Keywords:\n", cleaner.extract_keywords(sample_transcript))
