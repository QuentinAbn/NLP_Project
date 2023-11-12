from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.metrics import classification_report, confusion_matrix
import plotly.express as px
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

def preprocess_text(text):
    """
    This function is a complete preprocessing pipeline to make our dataset usable.
    """
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

def tokenize_data(X_train):
    """
    This function regroups some text preprocessing steps before using a NN model
    """
    # Tokenize text and keep only the 2000 most frequent words for computational efficiency
    tokenizer = Tokenizer(num_words=2000)
    tokenizer.fit_on_texts(X_train)
    # Converts text to a sequence of indices
    seq = tokenizer.texts_to_sequences(X_train)
    # Complete the sequences with zeros if needed so they all have the same length
    seq = pad_sequences(seq, maxlen=200)

    return tokenizer, seq


class Model:
    def __init__(self, X, y, model_architecture, vectorizer, random_seed=42, test_size=0.2) -> None:
        self.X = X
        self.y = y
        self.model_instance = model_architecture
        self.vectorizer = vectorizer
        self.random_seed = random_seed
        self.test_size = test_size
        # Define a pipeline
        self.pipeline = Pipeline([("Vectorizer", self.vectorizer), ("Model_Architecture", self.model_instance)])
        # Train test & split 
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=self.test_size, random_state=self.random_seed)
    
    def fit(self):
        # fit self.pipeline to the training data
        self.pipeline.fit(self.X_train, self.y_train)

    def predict(self):
        return self.pipeline.predict(self.X_test)

    
    def predict_proba(self):
        return self.pipeline.predict_proba(self.X_test)


    def results_report(self, class_labels):
        """
        This function creates reports containing informations on the performances of a model.
        It also prints a confusion matrix.
        """
        # Print a classification report to have informations a different metrics
        print(classification_report(self.y_test, self.predict(), target_names=class_labels))
        # Create a confusion matrix
        confusion_matrix_plot2 = confusion_matrix(self.y_test, self.predict())
        # Plot parameters of the confusion matrix
        fig = px.imshow(confusion_matrix_plot2, 
            text_auto=True, 
            title="Confusion Matrix", width=1000, height=800,
            labels=dict(x="Predicted", y="True Label"),
            x=class_labels,
            y=class_labels,
            color_continuous_scale='Blues'
            )
        # Plot the matrix
        fig.show()

    def fit_gridsearch(self, parameters):
        """
        This function realises a grid search to find the best parameters for the model.
        """
        # Define the grid search
        self.grid_search = GridSearchCV(self.pipeline, parameters, cv=5, n_jobs=-1)
        # Apply the grid search to the data
        self.grid_search.fit(self.X_train, self.y_train)


        