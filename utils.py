from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.metrics import classification_report, confusion_matrix
import plotly.express as px

def preprocess_text(text):
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))

    # Tokenization
    tokens = word_tokenize(text)
    # Remove punctuation and convert to lowercase
    words = [word.lower() for word in tokens if word.isalnum()]
    # Remove stopwords
    words = [word for word in words if word not in stop_words]
    # Lemmatization using WordNetLemmatizer
    words = [lemmatizer.lemmatize(word) for word in words]
    
    return ' '.join(words)


def results_report(y_true, y_pred, class_labels):

    confusion_matrix_kwargs = dict(
        text_auto=True, 
        title="Confusion Matrix", width=250, height=200,
        labels=dict(x="Predicted", y="True Label"),
        x=class_labels,
        y=class_labels,
        color_continuous_scale='Blues')
    
    print(classification_report(y_true, y_pred, target_names=class_labels))
    confusion_matrix_plot = confusion_matrix(y_true, y_pred)
    fig = px.imshow(confusion_matrix_plot, **confusion_matrix_kwargs)
    fig.show()