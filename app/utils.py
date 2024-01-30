import torch
import torch.nn as nn
import math

class LSTMLanguageModel(nn.Module):

    def __init__(self, vocab_size, emb_dim, hid_dim, num_layers, dropout_rate):

        super().__init__()
        self.num_layers = num_layers # defining lstm (how many layers of LSTM)
        self.hid_dim    = hid_dim    # vector size
        self.emb_dim    = emb_dim    
        
        self.embedding  = nn.Embedding(vocab_size, emb_dim) # input the text > get_embedded > sent to LSTM (vectorized)
                                                            # word -> embedding(vectorized)
        
        self.lstm       = nn.LSTM(emb_dim, hid_dim, num_layers=num_layers, dropout=dropout_rate, batch_first=True) # dropout connect -> drop weights between LSTM
                                                                                                                   # in paper
        
        # seq length -> 

        self.dropout    = nn.Dropout(dropout_rate) # after certain process

        # hidden dim to vocab size to softmax 
        self.fc         = nn.Linear(hid_dim, vocab_size) # prediction head

        self.init_weights()

    # optionally # inital weight with range
    def init_weights(self):
        # from the paper
        # by bounding them into specific range, the weight doesn't go too big
        
        init_range_emb   = 0.1 
        init_range_other = 1/math.sqrt(self.hid_dim) 
        self.embedding.weight.data.uniform_(-init_range_emb, init_range_other) 
        self.fc.weight.data.uniform_(-init_range_other, init_range_other)
        self.fc.bias.data.zero_() # bias is not effecting a lot, then zero

        for i in range(self.num_layers):
            self.lstm.all_weights[i][0] = torch.FloatTensor(self.emb_dim, self.hid_dim).uniform_(-init_range_other, init_range_other) #We #work with x
            self.lstm.all_weights[i][1] = torch.FloatTensor(self.hid_dim, self.hid_dim).uniform_(-init_range_other, init_range_other) #Wh #work with previous h
    
    # will be called in training (hidden,cell)
    def init_hidden(self, batch_size, device):
        hidden = torch.zeros(self.num_layers, batch_size, self.hid_dim).to(device) # to take fully control of hidden 
        cell   = torch.zeros(self.num_layers, batch_size, self.hid_dim).to(device) 
        return hidden, cell
           
    def detach_hidden(self, hidden):
        hidden, cell = hidden 
        hidden = hidden.detach() #not to be used for gradient computation
        cell   = cell.detach()
        return hidden, cell
        
    def forward(self, src, hidden): 
        #src: [batch_size, seq len]
        embedding  = self.dropout(self.embedding(src)) #harry potter is ... # can learn pattern # embedding dropout
        #embedding: [batch-size, seq len, emb dim]
        output, hidden = self.lstm(embedding, hidden) 
        #ouput: [batch size, seq len, hid dim] # 
        #hidden: [num_layers * direction, seq len, hid_dim]
        output     = self.dropout(output) # variation dropout is similar to dropout 

        #
        prediction = self.fc(output)
        #prediction: [batch_size, seq_len, vocab_size]
        return prediction, hidden # to carry foward hidden
    

def generate(prompt, max_seq_len, temperature, model, tokenizer, vocab, device, seed=None):
    if seed is not None:
        torch.manual_seed(seed)
    model.eval()
    tokens = tokenizer(prompt)
    indices = [vocab[t] for t in tokens]
    batch_size = 1
    hidden = model.init_hidden(batch_size, device)
    with torch.no_grad():
        for i in range(max_seq_len):
            src = torch.LongTensor([indices]).to(device)
            prediction, hidden = model(src, hidden)
            
            #prediction: [batch size, seq len, vocab size]
            #prediction[:, -1]: [batch size, vocab size] 
            #probability of last vocab
            
            # is = 0.3 on = 0.5 eat 0.2
            # is is is on on on on on eat eat >  sampling

            probs = torch.softmax(prediction[:, -1] / temperature, dim=-1)  
            prediction = torch.multinomial(probs, num_samples=1).item()    
            
            while prediction == vocab['<unk>']: #if it is unk, we sample again
                prediction = torch.multinomial(probs, num_samples=1).item()

            if prediction == vocab['<eos>']:    #if it is eos, we stop
                break

            indices.append(prediction) #autoregressive, thus output becomes input

    itos = vocab.get_itos()
    tokens = [itos[i] for i in indices]
    return tokens