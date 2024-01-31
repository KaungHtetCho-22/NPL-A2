# NPL-A1
NLP assignment from Asian Institute of Technology

# Table of Contents
1. [ID](#ID)
2. [Task1](#Task1)
3. [Task2](#Task2)
4. [Task3](#Task3)

## ID
Kaung Htet Cho (st124092)

## Task1
### Dataset Acquisition

Training_dataset   = Harry Potter 5 books (57435 rows)

Validation_dataset = Harry Potter 1 book (5897 rows)

Test_dataset       = Harry Potter 1 book (6589 rows)

Total_vocab        = 9803

The model has 36,879,947 trainable parameters

Downloaded JK-Rowling's Harry Potter book series (1 to 7) (.txt files) from https://github.com/ganesh-k13/shell/tree/master/test_search/www.glozman.com/TextPages?fbclid=IwAR0DN0fhdvHpZAYus94rMwnTlVFkAmb-D6JEyJCaPidAg8-BDkoIyS79WOs ,and uploaded to the Hugging Face repository https://huggingface.co/datasets/KaungHtetCho/Harry_Potter_LSTM in order to use the load_dataset function. 


## Task2
### Model training

### 1. Text preprocessing

The text is processed using a tokenizer from torchtext, dividing it into batches of 128 and limiting the sequence length later to 50 as inputs for LSTM model. Special tokens are included with <unk> at index 0 to represent unknown words and <eos> at index 1 to denote the end of a sequence. 

### 2. Model architecture and training Process

- epochs = 50
- device = Nvidia GeForce RTX 3060
- training perplexity = 35.901
- validation perplexity = 66.717
- testing perplexity = 84.753

The training process 
1. During training, a batch of text sequences is put to the model, which first converts the tokens into embeddings. 
2. Embeddings pass through the LSTM layers (nn.lstm), capturing shared weights between words. 
3. The model's output is then passed through a linear layer to generate predictions for the next tokens in the sequence. 
4. The hidden states are detached from the computation graph after each batch to prevent backpropagating through the entire history, to optimize memory usage. 


## Task3
### Web app documentation

The web app is located at localhost:8000 and features an input prompt. When a user types in any words, the preloaded LSTM model, configured with specific parameters (e.g., temperature = 1, max_seq_length = 100), generates token predictions. These generated tokens are then concatenated into a single string, separated by spaces. This concatenated string represents the final generated text.