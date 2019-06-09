Re-Implementation of EMNLP 2018 paper [Paraphrase Generation with Deep Reinforcement Learning](https://arxiv.org/pdf/1711.00279.pdf)

Still under construction.
________

## Updates:
- Created parse_data.py to process Quora dataset into tokens and .bin files (tf.Example) files for generator to read

## TODO:

### RbM-SL
1. To train evaluator:
    - Fix train.py to take in the data we want - no need to read_corpus, we just need to have pairs of (sent1, sent2, label) --> where label is 0 or 1
        - Embeddings: Use the ones from the paper, there should be a function to load them
2. To pre-train generator()
    - Fix batcher.py
        - modify text_generator() to extract question 1 and question 2 from tf.Example file IFF they have a positive label
        - modify fill_example_queue() --> Possibly just name change of (article, abstract)
3. Try generating one sentence with initial generator
4. Write the whole RL architecture:
    - Find a good monte carlo simulator library for this task
    - Function to compute value 
    - Make function to rescale Q value
    - Make function to update gradient
    
### RbM-iRL
1. Steps 1-3 from RbM-SL
2. Calculate gradient according to algorithm
3. Write method to train evaluator 
