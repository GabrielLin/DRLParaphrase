# Author: @dafirebanks 
# June 7, 2019
# Based on the code to process the CNN/DailyMail dataset found in: https://github.com/abisee/cnn-dailymail/blob/master/make_datafiles.py
#
# USAGE: python3 make_datafiles.py
# Reads in the Quora question pairs dataset, tokenizes them and creates tf.Example (.bin) files
# 
# BEFORE RUNNING:
# Modify train_tokenized_dir, val_tokenized_dir, test_tokenized_dir and finished_files_dir
# Download the Quora "train.csv" dataset and write down the path in fpath in main()

import sys
import os
import hashlib
import struct
import subprocess
import collections
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import pandas as pd
from tqdm import tqdm
from tensorflow.core.example import example_pb2


from nltk.parse import CoreNLPParser
parser = CoreNLPParser(url='http://localhost:8889')

VOCAB_SIZE = 5000

train_tokenized_dir = "~/quora/train_tokens"
val_tokenized_dir = "~/quora/val_tokens"
test_tokenized_dir = "~/quora/test_tokens"
finished_files_dir = "~/quora/finished_files/"

CHUNK_SIZE = 1000 # num examples per chunk, for the chunked data
chunks_dir = os.path.join(finished_files_dir, "chunked")

def process_data(fpath):
    # Move to dataframe for text usage
    df = pd.read_csv(fpath)
    print("Total number of pairs:", df.shape[0])
    
    # Convert all questions to string and X, y format for splitting
    train_q1 = [str(el) for el in train_df["question1"]]
    train_q2 = [str(el) for el in train_df["question2"]]
    X = [(q1, q2) for q1, q2 in zip(train_q1, train_q2)]
    y = list(train_df["is_duplicate"])
    
    # Sample splitting 60% train, 20% validation and 20% test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=1)
    
    return X_train, y_train, X_val, y_val, X_test, y_test

def tokenize_questions(questions):
    """Takes in a list of question pairs, returns a list of tokenized question pairs"""
    tokenized_qs = []
    for i in range(len(questions)):
        # Tokenizing both question1 and question2
        tokenized_qs.append([list(parser.tokenize(questions[i][0])), list(parser.tokenize(questions[i][1]))])
        if i % 10000 == 0:
            print(f"Tokenized {i} questions!")
            print(f"Q1: {questions[i][0]} \nQ2: {questions[i][1]}")
    return tokenized_qs

def store_tokens(questions, outdir):
    """ Stores a pair of tokenized questions separated by a new line"""
    
    fnum = 0
    for qs in questions: 
        with open(os.path.join(outdir, "qpair" + str(fnum) + ".tokens"), "w") as f:
            e1 = ' '.join(qs[0]).lower().strip()
            e2 = ' '.join(qs[1]).lower().strip()
            f.write(f"Q1: {e1} \nQ2: {e2}\n")
        
        fnum += 1

def write_to_bin(tokenized_qs, is_duplicate, outfile, makevocab=False):
    """ Creates bin files given pairs of tokenized questions, an outfile name and if applicable, creates a vocabulary file 
        
        @tokenized_qs: list of questions as tokenized strings [(question1_tokenized, question2_tokenized), (tokenized_pair_2), ...]
        @is_duplicate: label list, target variable
        @outfile: path as string"""
    
    if makevocab:
        vocab_counter = collections.Counter()
        
    with open(outfile, 'wb') as writer:
        
        for i in range(len(tokenized_qs)):
            tok_q1 = tokenized_qs[i][0]
            tok_q2 = tokenized_qs[i][1]
            target = str(is_duplicate[i])
            # TODO Important note: Max length of question is 20 words, I assume we clean up the symbols that are not question marks? OR we do this in the actual program
            # In the original program, there was no particular data cleaning, so I assume that this is done afterwards
            
            # Questions as strings: lowercase and strip them 
            q1 = ' '.join(tok_q1).lower().strip()
            q2 = ' '.join(tok_q2).lower().strip()
            
            # Write to tf.Example
            tf_example = example_pb2.Example()
            tf_example.features.feature['question1'].bytes_list.value.extend([q1.encode()])
            tf_example.features.feature['question2'].bytes_list.value.extend([q2.encode()])
            tf_example.features.feature['target'].bytes_list.value.extend([target.encode()])
            
            tf_example_str = tf_example.SerializeToString()
            str_len = len(tf_example_str)
            writer.write(struct.pack('q', str_len)) # write length of string
            writer.write(struct.pack('%ds' % str_len, tf_example_str)) # write string of length noted earlier
            
            
            # Make the vocab to write, if applicable
            if makevocab:
                tokens = q1 + q2
                tokens = [t.strip() for t in tokens] # strip
                tokens = [t for t in tokens if t != ""] # remove empty
                vocab_counter.update(tokens)
    
    print("Finished writing file %s\n" % outfile)
    
    # Write vocab to file
    if makevocab:
        print("Writing vocab file...")
        with open(os.path.join(finished_files_dir, "vocab"), 'w') as writer:
            for word, count in vocab_counter.most_common(VOCAB_SIZE):
                writer.write(word + ' ' + str(count) + '\n')
        print("Finished writing vocab file")

# Chunk functions for replications of paper, not sure if truly necessary?
def chunk_file(set_name, files_dir):
    in_file = files_dir % set_name
    reader = open(in_file, "rb")
    chunk = 0
    finished = False
    while not finished:
        chunk_fname = os.path.join(chunks_dir, '%s_%03d.bin' % (set_name, chunk)) # new chunk
        with open(chunk_fname, 'wb') as writer:
            for _ in range(CHUNK_SIZE):
                len_bytes = reader.read(8)
                if not len_bytes:
                    finished = True
                    break
                str_len = struct.unpack('q', len_bytes)[0]
                example_str = struct.unpack('%ds' % str_len, reader.read(str_len))[0]
                writer.write(struct.pack('q', str_len))
                writer.write(struct.pack('%ds' % str_len, example_str))
            chunk += 1


def chunk_all(files_dir):
    # Make a dir to hold the chunks
    if not os.path.isdir(chunks_dir):
        os.mkdir(chunks_dir)
    # Chunk the data
    for set_name in ['train', 'val', 'test']:
        print("Splitting %s data into chunks..." % set_name)
        chunk_file(set_name, files_dir)
    print("Saved chunked data in %s" % chunks_dir)


def main():
    """ 1. Get the train and test set paths, parse them into texts
        2. Tokenize all of them
        3. Create bin files 
        4. Chunk data 
    """

    # Alternative use: input file path from command line
    if len(sys.argv) != 1:
        print("USAGE: python make_datafiles.py")
        sys.exit()
        
    
    fpath = "~/quora/train.csv"

    # Move to dataframe for text usage
    X_train, y_train, X_val, y_val, X_test, y_test = process_data(fpath)

    # Create some new directories to store tokenized versions of the questions
    if not os.path.exists(train_tokenized_dir): 
        os.makedirs(train_tokenized_dir)
    if not os.path.exists(val_tokenized_dir): 
        os.makedirs(val_tokenized_dir)
    if not os.path.exists(test_tokenized_dir): 
        os.makedirs(test_tokenized_dir)
    if not os.path.exists(finished_files_dir): 
        os.makedirs(finished_files_dir)

    # Run stanford tokenizer on both sets, outputting to tokenized questions directories
    train_tokens = tokenize_questions(X_train)
    val_tokens = tokenize_questions(X_val)
    test_tokens = tokenize_questions(X_test)
    
    store_tokens(train_tokens, train_tokenized_dir)
    store_tokens(val_tokens, val_tokenized_dir)
    store_tokens(test_tokens, test_tokenized_dir)
    
    # Read the tokenized stories, do a little postprocessing then write to bin files
    write_to_bin(train_tokens, y_train, os.path.join(finished_files_dir, "train.bin"), makevocab=True)
    write_to_bin(val_tokens, y_val, os.path.join(finished_files_dir, "val.bin"))
    write_to_bin(test_tokens, y_test, os.path.join(finished_files_dir, "test.bin"))
    
    # Chunk the data. This splits each of train.bin, val.bin and test.bin into smaller chunks, each containing e.g. 1000 examples, and saves them in finished_files/chunks
    chunk_all('~/quora/finished_files/%s.bin')

