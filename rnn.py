import numpy as np
from sklearn.metrics import accuracy_score
import torch
import torch.nn as nn
from torch.nn import init
import torch.optim as optim
import math
import random
import os
import time
from tqdm import tqdm
import json
import string
from argparse import ArgumentParser
import pickle

unk = '<UNK>'
# Consult the PyTorch documentation for information on the functions used below:
# https://pytorch.org/docs/stable/torch.html
class RNN(nn.Module):
    def __init__(self, hidden_dim, embedding_layer):  # Add relevant parameters
        super(RNN, self).__init__()
        self.embedding = embedding_layer
        self.rnn = nn.RNN(
            input_size=embedding_layer.embedding_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            nonlinearity='tanh'
        )
        self.W = nn.Linear(hidden_dim, 5)
        self.softmax = nn.LogSoftmax(dim=1)
        self.loss = nn.NLLLoss()

    def compute_Loss(self, predicted_vector, gold_label):
        return self.loss(predicted_vector, gold_label)

    def forward(self, inputs):
        embedded = self.embedding(inputs)  #
        rnn_out, hidden = self.rnn(embedded)

        time_step_logits = self.W(rnn_out)
        summed_logits = torch.sum(time_step_logits, dim=0)
        predicted_vector = self.softmax(summed_logits)
        return predicted_vector


def load_data(train_data, val_data):
    with open(train_data) as training_f:
        training = json.load(training_f)
    with open(val_data) as valid_f:
        validation = json.load(valid_f)

    tra = []
    val = []
    for elt in training:
        tra.append((elt["text"].split(),int(elt["stars"]-1)))
    for elt in validation:
        val.append((elt["text"].split(),int(elt["stars"]-1)))
    return tra, val

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

def make_vocab(data):
    vocab = set()
    for document, _ in data:
        for word in document:
            vocab.add(word)
    return vocab

def load_test_data(test_data):
    with open(test_data) as test_f:
        test = json.load(test_f)

    tst = []
    for elt in test:
        tst.append((elt["text"].split(), int(elt["stars"]-1)))

    return tst



if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-hd", "--hidden_dim", type=int, required = True, help = "hidden_dim")
    parser.add_argument("-e", "--epochs", type=int, required = True, help = "num of epochs to train")
    parser.add_argument("--train_data", required = True, help = "path to training data")
    parser.add_argument("--val_data", required = True, help = "path to validation data")
    parser.add_argument("--test_data", required = True, help = "path to test data")
    parser.add_argument("--model_name", required=True, help="name the model")
    parser.add_argument('--do_test', action='store_true')
    args = parser.parse_args()

    print("========== Loading data ==========")
    train_data, valid_data = load_data(args.train_data, args.val_data) 
    vocab = make_vocab(train_data)
    vocab, word2index, index2word = make_indices(vocab)

    test_data = load_test_data(args.test_data)

    with open("word_embedding.pkl", "rb") as f:
        pretrained_embeddings = pickle.load(f)

    embedding_dim = len(next(iter(pretrained_embeddings.values())))
    embedding_matrix = np.random.randn(len(word2index), embedding_dim).astype(np.float32)

    for word, idx in word2index.items():
        if word in pretrained_embeddings:
            embedding_matrix[idx] = pretrained_embeddings[word]

    embedding_matrix_tensor = torch.tensor(embedding_matrix) 
    embedding_layer = nn.Embedding.from_pretrained(
        embedding_matrix_tensor,
        freeze=True
    )
    
    # X_data is a list of pairs (document, y); y in {0,1,2,3,4}

    # Think about the type of function that an RNN describes. To apply it, you will need to convert the text data into vector representations.
    # Further, think about where the vectors will come from. There are 3 reasonable choices:
    # 1) Randomly assign the input to vectors and learn better embeddings during training; see the PyTorch documentation for guidance
    # 2) Assign the input to vectors using pretrained word embeddings. We recommend any of {Word2Vec, GloVe, FastText}. Then, you do not train/update these embeddings.
    # 3) You do the same as 2) but you train (this is called fine-tuning) the pretrained embeddings further.
    # Option 3 will be the most time consuming, so we do not recommend starting with this

    save_path = f"./results/{args.model_name}/"
    os.makedirs(save_path, exist_ok=True)
    out_path = save_path + "test.out"

    with open(out_path, "a") as f:
        f.write("{:<10}{:<10}{:<15}{:<15}\n".format("Epoch", "Loss", "Train_acc", "Val_acc"))

    print("========== Vectorizing data ==========")
    model = RNN(hidden_dim=args.hidden_dim, embedding_layer=embedding_layer)
    # optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    word_embedding = pickle.load(open('./word_embedding.pkl', 'rb'))

    stopping_condition = False
    epoch = 0

    last_train_accuracy = 0
    last_validation_accuracy = 0

    while not stopping_condition and epoch < args.epochs:
        random.shuffle(train_data)
        model.train()
        # You will need further code to operationalize training, ffnn.py may be helpful
        print("Training started for epoch {}".format(epoch + 1))

        train_data = train_data
        correct = 0
        total = 0
        minibatch_size = 16
        N = len(train_data)

        loss_total = 0
        loss_count = 0

        for minibatch_index in tqdm(range(N // minibatch_size)):
            optimizer.zero_grad()
            loss = None

            for example_index in range(minibatch_size):
                input_words, gold_label = train_data[minibatch_index * minibatch_size + example_index]
                
                input_words = " ".join(input_words)
                input_words = input_words.translate(input_words.maketrans("", "", string.punctuation)).split()

                indices = [word2index.get(w.lower(), word2index['<UNK>']) for w in input_words]

                indices_tensor = torch.tensor(indices).view(len(indices), 1)
                output = model(indices_tensor)

                predicted_vector = torch.mean(output, dim=0)



                # Get loss
                example_loss = model.compute_Loss(output.view(1,-1), torch.tensor([gold_label]))

                # Get predicted label
                predicted_label = torch.argmax(predicted_vector)

                correct += int(predicted_label == gold_label)
                # print(predicted_label, gold_label)
                total += 1
                if loss is None:
                    loss = example_loss
                else:
                    loss += example_loss
                

            loss = loss / minibatch_size
            loss_total += loss.data
            loss_count += 1
            loss.backward()
            optimizer.step()

        with open(out_path, "a") as f:
            f.write("{:<10d}".format(epoch + 1))
            f.write("{:<10.4f}".format(loss))
            f.write("{:<15.4f}".format(correct / total))

        print(loss_total/loss_count)
        print("Training completed for epoch {}".format(epoch + 1))
        print("Training accuracy for epoch {}: {}".format(epoch + 1, correct / total))
        training_accuracy = correct/total


        model.eval()
        correct = 0
        total = 0
        random.shuffle(valid_data)
        print("Validation started for epoch {}".format(epoch + 1))

        valid_data = valid_data

        for input_words, gold_label in tqdm(valid_data):

            input_words = " ".join(input_words)
            input_words = input_words.translate(input_words.maketrans("", "", string.punctuation)).split()

            indices = [word2index.get(w.lower(), word2index['<UNK>']) for w in input_words]
            indices_tensor = torch.tensor(indices).view(len(indices), 1)

            output = model(indices_tensor)
            predicted_vector = torch.mean(output, dim=0)

            predicted_label = torch.argmax(predicted_vector)
            correct += int(predicted_label == gold_label)
            total += 1
           
        print("Validation completed for epoch {}".format(epoch + 1))
        print("Validation accuracy for epoch {}: {}".format(epoch + 1, correct / total))

        with open(out_path, "a") as f:
            f.write("{:<15.4f}".format(correct / total))
            f.write("\n")
        validation_accuracy = correct/total

        if validation_accuracy < last_validation_accuracy and training_accuracy > last_train_accuracy:
            stopping_condition=True
            print("Training done to avoid overfitting!")
            print("Best validation accuracy is:", last_validation_accuracy)
        else:
            last_validation_accuracy = validation_accuracy
            last_train_accuracy = training_accuracy

        epoch += 1

    if (args.do_test):
        with open(out_path, "a") as f:
            print("========== Saving Trained Model ==========")
            save_file = os.path.join(save_path, f"{args.model_name}")
            torch.save(model.state_dict(), save_file)

            print("========== Testing Model ==========")
            N = len(test_data)
            y_true = []
            y_pred = []
            random.shuffle(test_data)

            with torch.no_grad():
                model.eval()
                for minibatch_index in tqdm(range(N // minibatch_size)):
                    for example_index in range(minibatch_size):
                        input_vector, gold_label = test_data[minibatch_index * minibatch_size + example_index]
                        
                        cleaned = " ".join(input_words)
                        cleaned = cleaned.translate(cleaned.maketrans("", "", string.punctuation)).split()

                        indices = [word2index.get(w.lower(), word2index['<UNK>']) for w in cleaned]
                        indices_tensor = torch.tensor(indices).view(len(indices), 1)

                        output = model(indices_tensor)
                        predicted_vector = torch.mean(output, dim=0)

                        predicted_label = torch.argmax(predicted_vector).item()
                        y_pred.append(predicted_label)
                        y_true.append(gold_label)

            test_accuracy = accuracy_score(y_true, y_pred)
            print(y_true)
            print(y_pred)
            result = f"Test accuracy for {args.model_name}: {test_accuracy}"
            print(result)
            f.write("\n"+result)