import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
import time
from tqdm import tqdm
import json
from argparse import ArgumentParser
import pandas as pd

unk = '<UNK>'

# Define the Feedforward Neural Network (FFNN) class
class FFNN(nn.Module):
    def __init__(self, input_dim, h):
        super(FFNN, self).__init__()
        self.h = h
        self.W1 = nn.Linear(input_dim, h)
        self.activation = nn.ReLU()
        self.output_dim = 5
        self.W2 = nn.Linear(h, self.output_dim)
        self.softmax = nn.LogSoftmax(dim=0)
        self.loss = nn.NLLLoss()

    def compute_Loss(self, predicted_vector, gold_label):
        return self.loss(predicted_vector, gold_label)

    def forward(self, input_vector):
        hidden_layer = self.activation(self.W1(input_vector))
        output_layer = self.W2(hidden_layer)
        predicted_vector = self.softmax(output_layer)
        return predicted_vector

# Function to create the vocabulary
def make_vocab(data):
    vocab = set()
    for document, _ in data:
        for word in document:
            vocab.add(word)
    return vocab

# Function to create word-to-index and index-to-word mappings
def make_indices(vocab):
    vocab_list = sorted(vocab)
    vocab_list.append(unk)
    word2index = {}
    index2word = {}
    for index, word in enumerate(vocab_list):
        word2index[word] = index
        index2word[index] = word
    vocab.add(unk)
    return vocab, word2index, index2word

# Function to convert data into vector representations
def convert_to_vector_representation(data, word2index):
    vectorized_data = []
    for document, y in data:
        vector = torch.zeros(len(word2index))
        for word in document:
            index = word2index.get(word, word2index[unk])
            vector[index] += 1
        vectorized_data.append((vector, y))
    return vectorized_data

# Function to load training, validation, and test data
def load_data(train_data, val_data, test_data):
    with open(train_data) as training_f:
        training = json.load(training_f)
    with open(val_data) as valid_f:
        validation = json.load(valid_f)
    with open(test_data) as test_f:
        test = json.load(test_f)

    tra = [(elt["text"].split(), int(elt["stars"] - 1)) for elt in training]
    val = [(elt["text"].split(), int(elt["stars"] - 1)) for elt in validation]
    tst = [(elt["text"].split(), int(elt["stars"] - 1)) for elt in test]

    return tra, val, tst

# Function to test the model
def test_model(model, test_data):
    model.eval()
    correct_test = 0
    total_test = 0
    for input_vector, gold_label in tqdm(test_data):
        with torch.no_grad():
            predicted_vector = model(input_vector)
        predicted_label = torch.argmax(predicted_vector)
        correct_test += int(predicted_label == gold_label)
        total_test += 1

    test_acc = correct_test / total_test
    print(f"Test accuracy: {test_acc:.2f}")
    return test_acc

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-hd", "--hidden_dim", type=int, required=True, help="hidden_dim")
    parser.add_argument("-e", "--epochs", type=int, required=True, help="num of epochs to train")
    parser.add_argument("--train_data", required=True, help="path to training data")
    parser.add_argument("--val_data", required=True, help="path to validation data")
    parser.add_argument("--test_data", required=True, help="path to test data")
    parser.add_argument('--do_train', action='store_true')
    args = parser.parse_args()

    random.seed(42)
    torch.manual_seed(42)

    # Load data
    print("========== Loading data ==========")
    train_data, valid_data, test_data = load_data(args.train_data, args.val_data, args.test_data)
    vocab = make_vocab(train_data)
    vocab, word2index, index2word = make_indices(vocab)

    # Vectorize data
    print("========== Vectorizing data ==========")
    train_data = convert_to_vector_representation(train_data, word2index)
    valid_data = convert_to_vector_representation(valid_data, word2index)
    test_data = convert_to_vector_representation(test_data, word2index)

    # Initialize the model and optimizer
    model = FFNN(input_dim=len(vocab), h=args.hidden_dim)
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

    print("========== Training for {} epochs ==========")
    epoch_results = []

    for epoch in range(args.epochs):
        model.train()
        loss = None
        correct_train = 0
        total_train = 0
        start_time = time.time()
        random.shuffle(train_data)
        minibatch_size = 16
        N = len(train_data)

        for minibatch_index in tqdm(range(N // minibatch_size)):
            optimizer.zero_grad()
            loss = None
            for example_index in range(minibatch_size):
                input_vector, gold_label = train_data[minibatch_index * minibatch_size + example_index]
                predicted_vector = model(input_vector)
                predicted_label = torch.argmax(predicted_vector)
                correct_train += int(predicted_label == gold_label)
                total_train += 1
                example_loss = model.compute_Loss(predicted_vector.view(1, -1), torch.tensor([gold_label]))
                if loss is None:
                    loss = example_loss
                else:
                    loss += example_loss
            loss = loss / minibatch_size
            loss.backward()
            optimizer.step()

        train_acc = correct_train / total_train
        print(f"Training accuracy for epoch {epoch + 1}: {train_acc:.2f}")

        model.eval()
        correct_val = 0
        total_val = 0

        for input_vector, gold_label in valid_data:
            with torch.no_grad():
                predicted_vector = model(input_vector)
            predicted_label = torch.argmax(predicted_vector)
            correct_val += int(predicted_label == gold_label)
            total_val += 1

        val_acc = correct_val / total_val
        print(f"Validation accuracy for epoch {epoch + 1}: {val_acc:.2f}")

        epoch_results.append([epoch + 1, train_acc, val_acc])

    # Test the model and get test accuracy
    test_acc = test_model(model, test_data)

    # Add test accuracy to the DataFrame and save it
    df = pd.DataFrame(epoch_results, columns=["Epoch", "Train Accuracy", "Validation Accuracy"])
    df["Test Accuracy"] = test_acc
    csv_filename = f"{args.epochs}_epochs_{args.hidden_dim}_hidden_dim.csv"
    df.to_csv(csv_filename, index=False)
    print(f"Results saved to {csv_filename}")
