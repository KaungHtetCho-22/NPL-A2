# NPL-A1
NLP assignment from Asian Institute of Technology

# Table of Contents
1. [ID](#ID)
2. [Task1](#Task1)
3. [Task2](#Task2)
4. [Task3](#Task3)

## ID
Kaung Htet Cho (st124092)

## Task1 (Dataset Acquisition)

Training_dataset   = Harry Potter 5 books (57435 rows)

Validation_dataset = Harry Potter 1 book (5897 rows)

Test_dataset       = Harry Potter 1 book (6589 rows)

Total_vocab        = 9803

The model has 36,879,947 trainable parameters

Downloaded JK-Rowling's Harry Potter book series (1 to 7) (.txt files) from https://github.com/ganesh-k13/shell/tree/master/test_search/www.glozman.com/TextPages?fbclid=IwAR0DN0fhdvHpZAYus94rMwnTlVFkAmb-D6JEyJCaPidAg8-BDkoIyS79WOs ,and uploaded to the Hugging Face repository https://huggingface.co/datasets/KaungHtetCho/Harry_Potter_LSTM in order to use the load_dataset function. 


## Task2 (Model training)

### 1. Text preprocessing

The text is processed using a tokenizer from torchtext, dividing it into batches of 128 and limiting the sequence length to 50 for LSTM model. Special tokens are included with <unk> at index 0 to represent unknown words and <eos> at index 1 to denote the end of a sequence. 

### 2. Model architecture and training Process



## Task3 (Web app documentation)


