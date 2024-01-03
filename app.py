# from flask import Flask, render_template, request, jsonify


# from transformers import AutoModelForCausalLM, AutoTokenizer
# import torch


# # tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
# # model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-medium")

# tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
# model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-medium")

# app = Flask(__name__)

# @app.route("/")
# def index():
#     return render_template('chat.html')


# @app.route("/get", methods=["GET", "POST"])
# def chat():
#     msg = request.form["msg"]
#     input = msg
#     return get_Chat_response(input)


# def get_Chat_response(text):

#     # Let's chat for 5 lines
#     for step in range(5):
#         # encode the new user input, add the eos_token and return a tensor in Pytorch
#         new_user_input_ids = tokenizer.encode(str(text) + tokenizer.eos_token, return_tensors='pt')

#         # append the new user input tokens to the chat history
#         bot_input_ids = torch.cat([chat_history_ids, new_user_input_ids], dim=-1) if step > 0 else new_user_input_ids

#         # generated a response while limiting the total chat history to 1000 tokens, 
#         chat_history_ids = model.generate(bot_input_ids, max_length=1000, pad_token_id=tokenizer.eos_token_id)

#         # pretty print last ouput tokens from bot
#         return tokenizer.decode(chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)


# if __name__ == '__main__':
#     app.run()


from flask import Flask, render_template, request, jsonify
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import os

app = Flask(__name__)

def load_or_download_model(local_dir, model_name):
    # Check if the model and tokenizer are already downloaded
    if not os.path.exists(os.path.join(local_dir, 'config.json')) or not os.path.exists(os.path.join(local_dir, 'pytorch_model.bin')):
        print("Downloading and saving model...")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name)
        tokenizer.save_pretrained(local_dir)
        model.save_pretrained(local_dir)
    else:
        print("Loading model from local directory...")
        tokenizer = AutoTokenizer.from_pretrained(local_dir)
        model = AutoModelForCausalLM.from_pretrained(local_dir)

    return tokenizer, model

# Define model name and local directory
model_name = "microsoft/DialoGPT-medium"
# model_name = "tiiuae/falcon-40b"
# local_dir = "/home/ubuntu/Multi-Modal-Chat-Bot/Falcon-40B-Model"
local_dir = "/home/ubuntu/Multi-Modal-Chat-Bot/DialoGPT-medium"

# Load or download model and tokenizer
tokenizer, model = load_or_download_model(local_dir, model_name)

@app.route("/")
def index():
    return render_template('chat.html')

@app.route("/get", methods=["GET", "POST"])
def chat():
    msg = request.form["msg"]
    return get_chat_response(msg)

def get_chat_response(text):
    new_user_input_ids = tokenizer.encode(text + tokenizer.eos_token, return_tensors='pt')
    chat_history_ids = model.generate(new_user_input_ids, max_length=100, pad_token_id=tokenizer.eos_token_id)
    response = tokenizer.decode(chat_history_ids[:, new_user_input_ids.shape[-1]:][0], skip_special_tokens=True)
    return jsonify({"response": response})

if __name__ == '__main__':
    app.run(debug=True)  # Turn off debug mode in production
