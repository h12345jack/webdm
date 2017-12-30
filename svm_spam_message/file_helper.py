import os

SVM_SPAM_MESSAGE_DIR = os.path.abspath(os.path.dirname(__file__))

def get_path(relative_path):
    return os.path.join(SVM_SPAM_MESSAGE_DIR, relative_path)

TRAINING_DATA_PATH = get_path('data/training_data.txt')
TEST_DATA_PATH = get_path('data/test_data.txt')

# tools path
TFIDF_VZ_PATH = get_path('tool/tfidf_vectorizer.pkl')

# data path
TRAINING_FEATURE_MATRIX_PATH = get_path('data/training_feature_matrix.pkl')
TRAINING_LABEL_PATH = get_path('data/training_labels.pkl')
TEST_FEATURE_MATRIX_PATH = get_path('data/test_feature_matrix.pkl')
TEST_LABEL_PATH = get_path('data/test_labels.pkl')


# model path
SVM_MODEL_PATH = get_path('model/svm.pkl')

