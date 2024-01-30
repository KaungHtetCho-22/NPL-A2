# import libraries
import torch
import pickle
import torch.nn.functional as F
from flask import Flask, render_template, request
from utils import LSTMLanguageModel, generate

# load the data
Data         = pickle.load(open('./models/data.pkl', 'rb'))
vocab_size   = Data['vocab_size']
emb_dim      = Data['emb_dim']
hid_dim      = Data['hid_dim']
num_layers   = Data['num_layers']
dropout_rate = Data['dropout_rate']
tokenizer    = Data['tokenizer']
vocab        = Data['vocab']

# load the model
lm_lstm = LSTMLanguageModel(vocab_size, emb_dim, hid_dim, num_layers, dropout_rate)
lm_lstm.load_state_dict(torch.load('./models/best-val-lstm_lm.pt', map_location=torch.device('cpu') ))
lm_lstm.eval()

app = Flask(__name__, static_url_path = '/static')

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'GET':
        return render_template('index.html', query = '', generated_text = '')

    if request.method == 'POST':
        prompt = request.form['text_input'].strip()  

        if not prompt:
            # Handle empty prompt: Show an error message or just reload the page
            error_msg = "Please enter a text prompt."
            return render_template('index.html', query = '', generated_text = '', error = error_msg)

        max_seq_len = 100  
        temperature = 1
        predefined_seed = 123

        # Generate text
        generated_tokens = generate(prompt, max_seq_len, temperature, lm_lstm, tokenizer, vocab, device='cpu', seed = predefined_seed)
        generated_text = ' '.join(generated_tokens)

        return render_template('index.html', query = prompt, generated_text = generated_text)

port_number = 8000

if __name__ == '__main__':
    app.run(debug = True, host='0.0.0.0', port=port_number)
    

