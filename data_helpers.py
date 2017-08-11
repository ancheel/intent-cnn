import numpy as np
import re
from sklearn.datasets import fetch_20newsgroups
from sklearn.datasets import load_files


def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()


def batch_iter(data, batch_size, num_epochs, shuffle=False):
    """
    Generates a batch iterator for a dataset.
    """
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int((len(data)-1)/batch_size) + 1
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]

def get_datasets_20newsgroup(subset='train', categories=None, shuffle=True, random_state=42):
    """
        Retrieve data from 20 newsgroups
        :param subset: train, test or all
        :param categories: List of newsgroup name
        :param shuffle: shuffle the list or not
        :param random_state: seed integer to shuffle the dataset
        :return: data and labels of the newsgroup
        """
    datasets = fetch_20newsgroups(subset=subset, categories=categories, shuffle=shuffle, random_state=random_state)
    #print(datasets)
    return datasets

def get_datasets_abstract(dir):
    print('+++++++++abstract+++++++++')
    #dir = 'datatrain/'
    task_examples = list(open(dir + 'task_data_file', "r").readlines())
    task_examples = [s.strip() for s in task_examples]
    model_examples = list(open(dir + 'model_data_file', "r").readlines())
    model_examples = [s.strip() for s in model_examples]
    dataset_examples = list(open(dir + 'dataset_data_file', "r").readlines())
    dataset_examples = [s.strip() for s in dataset_examples]
    result_examples = list(open(dir + 'result_data_file', "r").readlines())
    result_examples = [s.strip() for s in result_examples]
    other_examples = list(open(dir + 'other_data_file', "r").readlines())
    other_examples = [s.strip() for s in other_examples]
    task_model_examples = list(open(dir + 'task_model_file', "r").readlines())
    task_model_examples = [s.strip() for s in task_model_examples]
    data_result_examples = list(open(dir + 'data_result_file', "r").readlines())
    data_result_examples = [s.strip() for s in data_result_examples]
    all_examples = list(open(dir + 'all_file', "r").readlines())
    all_examples = [s.strip() for s in all_examples]


    data_train = dict()
    data_train['data'] = all_examples
    target = [0 for x in all_examples]
    data_train['target'] = target
    data_train['target_names'] = ['all_examples']
    return data_train

def get_datasets_intents(dir):

    expert_examples = list(open(dir + 'expert.txt', "r").readlines())
    expert_examples = [s.strip() for s in expert_examples]
    paper_examples = list(open(dir + 'paper.txt', "r").readlines())
    paper_examples = [s.strip() for s in paper_examples]
    conference_examples = list(open(dir + 'conference.txt', "r").readlines())
    conference_examples = [s.strip() for s in conference_examples]

    data_train = dict()
    data_train['data'] = expert_examples + paper_examples + conference_examples
    target = [0 for x in expert_examples] + [1 for x in paper_examples] + [2 for x in conference_examples]
    data_train['target'] = target
    data_train['target_names'] = ['expert', 'paper', 'conference']

    return data_train

def get_datasets_intentst(dir):

    expert_examples = list(open(dir + 'expertt.txt', "r").readlines())
    expert_examples = [s.strip() for s in expert_examples]
    paper_examples = list(open(dir + 'papert.txt', "r").readlines())
    paper_examples = [s.strip() for s in paper_examples]
    conference_examples = list(open(dir + 'conferencet.txt', "r").readlines())
    conference_examples = [s.strip() for s in conference_examples]

    data_train = dict()
    data_train['data'] = expert_examples + paper_examples + conference_examples
    target = [0 for x in expert_examples] + [1 for x in paper_examples] + [2 for x in conference_examples]
    data_train['target'] = target
    data_train['target_names'] = ['expert', 'paper', 'conference']

    return data_train


def get_datasets_mrpolarity(positive_data_file, negative_data_file):
    """
    Loads MR polarity data from files, splits the data into words and generates labels.
    Returns split sentences and labels.
    """
    # Load data from files
    positive_examples = list(open(positive_data_file, "r").readlines())
    positive_examples = [s.strip() for s in positive_examples]
    negative_examples = list(open(negative_data_file, "r").readlines())
    negative_examples = [s.strip() for s in negative_examples]

    datasets = dict()
    datasets['data'] = positive_examples + negative_examples
    target = [0 for x in positive_examples] + [1 for x in negative_examples]
    datasets['target'] = target
    datasets['target_names'] = ['positive_examples', 'negative_examples']
    return datasets


def get_datasets_localdata(container_path=None, categories=None, load_content=True,
                       encoding='utf-8', shuffle=True, random_state=42):
    """
    Load text files with categories as subfolder names.
    Individual samples are assumed to be files stored a two levels folder structure.
    :param container_path: The path of the container
    :param categories: List of classes to choose, all classes are chosen by default (if empty or omitted)
    :param shuffle: shuffle the list or not
    :param random_state: seed integer to shuffle the dataset
    :return: data and labels of the dataset
    """
    datasets = load_files(container_path=container_path, categories=categories,
                          load_content=load_content, shuffle=shuffle, encoding=encoding,
                          random_state=random_state)
    return datasets


def load_data_labels(datasets):
    """
    Load data and labels
    :param datasets:
    :return:
    """
    # Split by words
    x_text = datasets['data']
    x_text = [clean_str(sent) for sent in x_text]
    # Generate labels
    labels = []
    for i in range(len(x_text)):
        label = [0 for j in datasets['target_names']]
        label[datasets['target'][i]] = 1
        labels.append(label)
    y = np.array(labels)
    return [x_text, y]


def load_embedding_vectors_word2vec(vocabulary, filename, binary):
    # load embedding_vectors from the word2vec
    encoding = 'utf-8'
    with open(filename, "rb") as f:
        header = f.readline()
        vocab_size, vector_size = map(int, header.split())
        # initial matrix with random uniform
        embedding_vectors = np.random.uniform(-0.25, 0.25, (len(vocabulary), vector_size))
        if binary:
            binary_len = np.dtype('float32').itemsize * vector_size
            for line_no in range(vocab_size):
                word = []
                while True:
                    ch = f.read(1)
                    if ch == b' ':
                        break
                    if ch == b'':
                        raise EOFError("unexpected end of input; is count incorrect or file otherwise damaged?")
                    if ch != b'\n':
                        word.append(ch)
                word = str(b''.join(word), encoding=encoding, errors='strict')
                idx = vocabulary.get(word)
                if idx != 0:
                    embedding_vectors[idx] = np.fromstring(f.read(binary_len), dtype='float32')
                else:
                    f.seek(binary_len, 1)
        else:
            for line_no in range(vocab_size):
                line = f.readline()
                if line == b'':
                    raise EOFError("unexpected end of input; is count incorrect or file otherwise damaged?")
                parts = str(line.rstrip(), encoding=encoding, errors='strict').split(" ")
                if len(parts) != vector_size + 1:
                    raise ValueError("invalid vector on line %s (is this really the text format?)" % (line_no))
                word, vector = parts[0], list(map('float32', parts[1:]))
                idx = vocabulary.get(word)
                if idx != 0:
                    embedding_vectors[idx] = vector
        f.close()
        return embedding_vectors


def load_embedding_vectors_glove(vocabulary, filename, vector_size):
    # load embedding_vectors from the glove
    # initial matrix with random uniform
    embedding_vectors = np.random.uniform(-0.25, 0.25, (len(vocabulary), vector_size))
    f = open(filename)
    for line in f:
        values = line.split()
        word = values[0]
        vector = np.asarray(values[1:], dtype="float32")
        idx = vocabulary.get(word)
        if idx != 0:
            embedding_vectors[idx] = vector
    f.close()
    return embedding_vectors

def load_char_embedding(vocabulary):
    alphabet = "abcdefghijklmnopqrstuvwxyz0123456789-,;.!?:'\"/\\|_@#$%^&*~`+-=<>()[]{}"
    one_hot = np.eye(26, dtype='int64')
    embedding_vectors = np.random.uniform(-0.25, 0.25, (len(vocabulary), 26))
    zero = np.zeros(26)
    for i in range(len(alphabet)):
        idx = vocabulary.get(alphabet[i])
        if i < 26:
            embedding_vectors[idx] = one_hot[i]
        else:
            embedding_vectors[idx] = zero
    return embedding_vectors

