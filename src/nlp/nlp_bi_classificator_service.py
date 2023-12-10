import numpy as np
import pandas as pd
import unittest
import logging
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import os
import ssl
from sklearn.feature_extraction.text import TfidfVectorizer
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
import torch
from torch.optim import Adam
from sklearn.preprocessing import LabelEncoder
import numpy as np
import torch
import torch.nn as nn
from torchtext.vocab import GloVe
from gensim.models import Word2Vec
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, matthews_corrcoef
import numpy as np
import torch
import pickle

# Create a separate logger for debug file logging
log_metrics = logging.getLogger('debugLogger')
log_metrics.setLevel(logging.DEBUG)  # Set to DEBUG level
file_handler = logging.FileHandler('./logs/nlp_metrics.log')
file_handler.setLevel(logging.DEBUG)  # Set to DEBUG level
file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
log_metrics.addHandler(file_handler)

# Configure basic logging to console for other logs
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer() 
vectorizer = TfidfVectorizer()
label_encoder = LabelEncoder()

# Settings for the nltk vocabulary. Set the path to the downloaded certificate file
cert_path = '/path/to/cacert.pem'  # Replace with the actual path to the cacert.pem file
# Set the SSL_CERT_FILE environment variable
os.environ['SSL_CERT_FILE'] = cert_path
# If necessary, you can also adjust the default SSL context in Python
ssl._create_default_https_context = ssl._create_unverified_context
nltk.download("english")
nltk.download('stopwords')
nltk.download('wordnet')

# Assuming you have defined the size of your vocabulary and the embedding dimension
vocab_size = 10000  # for example, the size of your vocabulary
cont_words_in_sentence = 25
dim_size_vector=25
glove = GloVe(name="twitter.27B", dim=dim_size_vector)
# Initialize the embedding layer
#word_embeddings = nn.Embedding(num_embeddings=vocab_size, embedding_dim=vector_max_len)
            
# 1 DATA PREPROCRESSING 
def read_dataset(file_path: str) -> pd.DataFrame:
    return pd.read_csv(file_path)

def remove_stopwords(text: str) -> str:    
    return ' '.join([word for word in text.split() if word not in stop_words])

def clean_txt(txt: str) -> str:
    # Remove punctuation and numbers
    if len(txt) < 3:
        return ''
    txt = re.sub('[^a-zA-Z]', ' ', txt)
    # Convert to lower case
    txt = txt.lower()
    return txt.strip()

def lemmas_txt(txt: str) -> str:
    return ' '.join([lemmatizer.lemmatize(word) for word in txt.split()])

def create_custom_embedding(custom_string, vector_max_len):
    '''
    OOV words (Out-of-vocabulary (OOV) words) can be converted to the vector with custom embedding  
    Trains a Word2Vec model on the provided string to create custom embeddings for OOV words
    and integrates them into an existing PyTorch embedding layer.
    '''
    sentences = [custom_string.split()]
    word2vec_model = Word2Vec(sentences, vector_size=vector_max_len, min_count=1, window=5, sg=1)
    return torch.tensor(word2vec_model.wv[custom_string], dtype=torch.float32)   

def vectorize_text(text):
    # Related to the vector representation 
    vectorized_text = []
    for word in text.split(' '):  # Assuming text is already tokenized into words
        if word in glove.stoi:
            vectorized_text.append(glove[word])
        else:            
            # OOV words (Out-of-vocabulary (OOV) words)
            vectorized_text.append(create_custom_embedding(word, dim_size_vector))
            #vectorized_text.append(torch.zeros(vector_size))  # Zero vector for unknown words
            
        
    # Padding, related to the amount of the words in the sentence 
    num_padding = cont_words_in_sentence - len(vectorized_text)
    if num_padding > 0:
        # Add padding if the sentence is shorter than max_len
        vectorized_text.extend([torch.zeros(cont_words_in_sentence) for _ in range(num_padding)])
    elif num_padding < 0:
        # Truncate the sentence if it is longer than max_len
        vectorized_text = vectorized_text[:cont_words_in_sentence]
    
    return vectorized_text

def encode_category(df: pd.DataFrame) -> pd.DataFrame:
    label_encoder = LabelEncoder()
    df['Category_Encoded'] = label_encoder.fit_transform(df['Category'])
    return df

def prepare_df_for_ml_training(df: pd.DataFrame):    
    df['Message'] = df['Message'].apply(clean_txt)
    df['Message'] = df['Message'].apply(lemmas_txt)
    # Replace empty strings with NaN
    df["Message"] = df["Message"].replace('', np.nan)
    # Drop rows with NaN in the specified column
    df = df.dropna(subset=["Message"])
    df = encode_category(df)
    return df

# 2 DataLoader preparation
class PredictionDataset(Dataset):
    def __init__(self, features):
        self.features = features
        logging.info(f"PredictionDataset features {self.features}") 
    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx]

class TrainTestSpamHamDataset(Dataset):
    def __init__(self, features_list, targets):
        # torch.cat function in PyTorch is used to concatenate a sequence of tensors along a specified dimension.        
        self.features = features_list
        self.targets = targets
    
    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.targets[idx]

def get_features_and_targets(df: pd.DataFrame):    
    targets = torch.tensor(df['Category_Encoded'].values, dtype=torch.long)
    logging.info(f"labels len: {targets.shape}")
    logging.info(f"labels data: {targets}")
    
    features = df['Message'].apply(vectorize_text)
    X_features_torch = list(map(lambda x: torch.from_numpy(np.stack(x)), features))
    
    logging.info(f"features len: {features.shape}")
    logging.info(f"features data: {features}")
    
    return X_features_torch, targets
    
def get_data_loaders(csv_file_path: str, batch_size: int = 32, test_size: float = 0.2) -> (DataLoader, DataLoader):
    # Read and preprocess the dataset
    df = pd.read_csv(csv_file_path) #, nrows=10)
    df = prepare_df_for_ml_training(df)

    # Split the data into training and testing sets
    train_df, test_df = train_test_split(df, test_size=test_size)

    # Create datasets
    train_features, train_targets = get_features_and_targets(train_df)
    test_features, test_targets = get_features_and_targets(test_df)
    logging.info(f"Train features len: {len(train_features)}")
    logging.info(f"Train_features[0][0].shape: {train_features[0][0].shape}")
    
    train_dataset = TrainTestSpamHamDataset(train_features, train_targets)
    test_dataset = TrainTestSpamHamDataset(test_features, test_targets)

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader

# MODEL preparation
class SpamHamNN(nn.Module):
    """Some Information about MyModule"""
    def __init__(self, num_features):
        super().__init__()
        self.flatten = nn.Flatten()
        self.lin_relu_stack = nn.Sequential(            
            #features * 1D words vector len * 2D vector slices)
            nn.Linear(((num_features * cont_words_in_sentence * dim_size_vector)), 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 2))

    def forward(self, x):
        logging.info(f"forward => {x.shape}")
        x = self.flatten(x)
        logging.info(f"self.flatten x => {x.shape}")
        pred = self.lin_relu_stack(x)
        logging.info(f"Linear ReLu Stack 0 => {pred[0]}")
        return pred


def calculate_metrics(predictions, true_labels, probabilities, loss=None):
    accuracy = accuracy_score(true_labels, predictions)
    precision = precision_score(true_labels, predictions)
    recall = recall_score(true_labels, predictions)
    f1 = f1_score(true_labels, predictions)
    roc_auc = roc_auc_score(true_labels, probabilities)
    mcc = matthews_corrcoef(true_labels, predictions)

    log_metrics.debug(f"Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}")
    log_metrics.debug(f"ROC AUC: {roc_auc:.4f}")
    log_metrics.debug(f"MCC: {mcc:.4f}")
    if loss is not None:
        log_metrics.debug(f"Epoch Loss: {loss:.4f}")

    
def train_model(model: SpamHamNN, train_loader: DataLoader, learning_rate: float = 0.001, epochs: int = 10):
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=learning_rate)

    for epoch in range(epochs):
        log_metrics.debug(f"Epoch {epoch+1}-------------------------------")
        all_predictions = []
        all_true_labels = []
        all_probabilities = []  # For ROC AUC

        for batch, (X, y) in enumerate(train_loader):
            optimizer.zero_grad()
            pred = model(X)
            loss = criterion(pred, y)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            all_probabilities.extend(pred[:, 1].detach().numpy())  # Assuming binary classification
            all_predictions.extend(pred.argmax(1).tolist())
            all_true_labels.extend(y.tolist())

        # Calculate and log metrics at the end of the epoch
        calculate_metrics(all_predictions, all_true_labels, all_probabilities, loss.item())
        
    return model    


def test_model(model: SpamHamNN, test_loader: DataLoader):
    log_metrics.debug("Test Model -------------------------------")
    model.eval()  # Set the model to evaluation mode
    all_predictions = []
    all_true_labels = []
    all_probabilities = []  # For ROC AUC

    with torch.no_grad():  # No gradient calculation needed for evaluation
        for X, y in test_loader:
            pred = model(X)

            all_probabilities.extend(pred[:, 1].detach().numpy())  # Assuming binary classification
            all_predictions.extend(pred.argmax(1).tolist())
            all_true_labels.extend(y.tolist())

    # Calculate and log metrics
    calculate_metrics(all_predictions, all_true_labels, all_probabilities)


#TESTS TDD. 
class TestDataPreprocessing(unittest.TestCase):
    def test_read_dataset(self):
        data = read_dataset('./data/spam_data.csv')
        self.assertFalse(data.empty)    
        
    def test_clean_txt(self):
        mock_data = { 'Category': ['1'], 'Message': ["- * # 1 2 4 Hello WORLD"]}        
        mock_df = pd.DataFrame(mock_data)
        mock_df['Message'] = mock_df['Message'].apply(clean_txt)        
        self.assertTrue(mock_df['Message'][0] == "hello world")       
        
    def test_lemmas_txt(self):
        mock_data = { 'Category': ['1'], 
                     'Message': ["The children are playing"]}        
        mock_df = pd.DataFrame(mock_data)
        mock_df['Message'] = mock_df['Message'].apply(lemmas_txt)        
        self.assertTrue(mock_df['Message'][0] == "The child are playing")                  
       
    def test_str_to_vect_not_words(self):
        mock_df = pd.DataFrame({'Category': ['1'], 'Message': ["hello qwerr"]})
        mock_df['Message'] = mock_df['Message'].apply(vectorize_text) 
        logging.info(f"{mock_df['Message'][0][0]}")   
        logging.info(f"{mock_df['Message'][0][1]}")   
        logging.info(f"{mock_df['Message'][0][2]}")   
        self.assertTrue(mock_df['Message'][0].shape == (25, 25))                      
        
        
def run_tests():        
    suite = unittest.TestSuite()
    suite.addTest(TestDataPreprocessing('test_read_dataset'))
    suite.addTest(TestDataPreprocessing('test_clean_txt'))
    suite.addTest(TestDataPreprocessing('test_lemmas_txt'))
    suite.addTest(TestDataPreprocessing('test_str_to_vect_not_words'))
    
    # Run the test
    runner = unittest.TextTestRunner()
    runner.run(suite)
             

def load_model():    
    model = SpamHamNN(num_features=1) 
    model.load_state_dict(torch.load("/models/spam_ham_model"))
    model.eval()
    return model


def store_model_pkl(model):
    # Assuming 'model' is your PyTorch model
    with open("./models/spam_ham_model.pkl", "wb") as file:
        pickle.dump(model, file)
        
def load_model_pkl(path):
    with open(path, "rb") as file:
        model = pickle.load(file)        
    return model 

def load_pkl_and_pred():
    model = load_model_pkl("./models/spam_ham_model.pkl")
    model.eval()
    df = pd.read_csv("./data/spam_data.csv", nrows=10)
    df = prepare_df_for_ml_training(df)
    
    test_features, test_targets = get_features_and_targets(df)
   
    #x = test_features[0] 
    pred = PredictionDataset(test_features)
    dataLoader = DataLoader(pred)
   
    all_probabilities = []
    all_predictions = []
    for x in dataLoader:
        pred = model(x)
        all_probabilities.extend(pred[:, 1].detach().numpy())  # Assuming binary classification
        all_predictions.extend(pred.argmax(1).tolist())
        logging.info(f"Predicted: {pred}")
    

if __name__ == "__main__":
    #run_tests() 
    
    #TRAIN the model
    #train_loader, test_loader = get_data_loaders("./data/spam_data.csv", batch_size = 100, test_size = 0.2)    
    #model = SpamHamNN(num_features=1)    
    #model = train_model(model, train_loader, learning_rate=0.001, epochs= 10) 
    #test_model(model, test_loader)    
    #torch.save(model.state_dict(), "./models/spam_ham_model.pt")
    #store_model_pkl(model)
    
    #LOAD model and test
    load_pkl_and_pred()
    