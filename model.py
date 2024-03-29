# -*- coding: utf-8 -*-
"""ML_3.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1nZ51LlepKF1rdmOkCDMTur3MliWw4O2O
"""

import pandas as pd
import jieba

# 指定CSV文件的路徑
file_path_ZH = "train-ZH.csv"
file_path_TL = "train-TL.csv"
test_file_path_ZH = "test-ZH-nospace.csv"

# 使用pandas讀取CSV文件
data_ZH = pd.read_csv(file_path_ZH)
data_TL = pd.read_csv(file_path_TL)
test_data_ZH = pd.read_csv(test_file_path_ZH)

# data_TL和data_ZL包含了CSV文件中的data
data_ZH.rename(columns={"id": "id", "txt": "input"}, inplace=True)
data_TL.rename(columns={"id": "id", "txt": "output"}, inplace=True)
test_data_ZH.rename(columns={"id": "id", "txt": "input"}, inplace=True)
print(data_ZH)
print(data_TL)
print(test_data_ZH)

data_ZH['input'] = data_ZH['input'].apply(lambda x: x.split() if isinstance(x, str) else [])
data_TL['output'] = data_TL['output'].apply(lambda x: x.split() if isinstance(x, str) else [])
test_data_ZH['input'] = test_data_ZH['input'].apply(lambda x: list(jieba.cut(str(x))) if isinstance(x, str) else [])
print(data_ZH)
print(data_TL)
print(test_data_ZH)

# input
tokenized_data = data_ZH['input'].tolist()
input_vocab = {'<PAD>': 0, '<UNK>': 1, '<BOS>': 2, '<EOS>': 3}

for sentence in tokenized_data:
    for word in sentence:
        if word not in input_vocab:
            input_vocab[word] = len(input_vocab)

input_num = [[input_vocab.get(word, input_vocab['<UNK>']) for word in sentence] for sentence in tokenized_data]

# output
tokenized_data = data_TL['output'].tolist()
output_vocab = {'<PAD>': 0, '<UNK>': 1, '<BOS>': 2, '<EOS>': 3}

for sentence in tokenized_data:
    for word in sentence:
        if word not in output_vocab:
            output_vocab[word] = len(output_vocab)

output_num = [[output_vocab.get(word, output_vocab['<UNK>']) for word in sentence] for sentence in tokenized_data]

# test_input
tokenized_data = test_data_ZH['input'].tolist()
test_input_num = [[input_vocab.get(word, input_vocab['<UNK>']) for word in sentence] for sentence in tokenized_data]
for sequence in test_input_num:
    print(sequence)

import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F
from itertools import chain

# Convert the data to PyTorch tensors
input_tensor = pad_sequence([torch.tensor(seq) for seq in input_num], batch_first=True, padding_value=input_vocab['<PAD>'])
output_tensor = pad_sequence([torch.tensor(seq) for seq in output_num], batch_first=True, padding_value=output_vocab['<PAD>'])
test_input_tensor = pad_sequence([torch.tensor(seq) for seq in test_input_num], batch_first=True, padding_value=input_vocab['<PAD>'])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
input_tensor, output_tensor, test_input_tensor = input_tensor.to(device), output_tensor.to(device), test_input_tensor.to(device)


# Create a TensorDataset
dataset = TensorDataset(input_tensor, output_tensor)
train_dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

class TransformerEncoderLayerWithBatchFirst(nn.TransformerEncoderLayer):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu"):
        super(TransformerEncoderLayerWithBatchFirst, self).__init__(
            d_model, nhead, dim_feedforward, dropout, activation
        )
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)

# Define the Transformer model
class Transformer(nn.Module):
    def __init__(self, input_vocab_size, output_vocab_size, d_model=256, nhead=4, num_encoder_layers=3, num_decoder_layers=3):
        super(Transformer, self).__init__()

        # Define input embedding
        self.embedding = nn.Embedding(input_vocab_size, d_model)

        # Define Transformer encoder
        self.transformer_encoder = nn.TransformerEncoder(
            TransformerEncoderLayerWithBatchFirst(d_model, nhead),
            num_layers=num_encoder_layers
        )

        # Define output embedding
        self.decoder_embedding = nn.Embedding(output_vocab_size, d_model)

        # Define Transformer decoder
        self.transformer_decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d_model, nhead),
            num_layers=num_decoder_layers
        )
        # Define output layer
        self.fc_out = nn.Linear(d_model, output_vocab_size)

    def forward(self, src, tgt):
        # Embed the source and target sequences
        src = self.embedding(src)
        tgt = self.decoder_embedding(tgt)

        # Transformer encoder
        memory = self.transformer_encoder(src)

        # Transformer decoder
        output = self.transformer_decoder(tgt, memory)

        # Linear layer for output
        output = self.fc_out(output)

        return output

# Instantiate the model
input_vocab_size = len(input_vocab)
output_vocab_size = len(output_vocab)
model = Transformer(input_vocab_size, output_vocab_size)
model.to(device)

# Define loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
epochs = 1
for epoch in range(epochs):
    for batch_input, batch_output in train_dataloader:
        batch_input, batch_output = batch_input.to(device), batch_output.to(device)

        # Forward pass
        output = model(batch_input, batch_output)

        # 建立反向的vocab
        reverse_output_vocab = {idx: word for word, idx in output_vocab.items()}
        #print("2", reverse_output_vocab)

        flat_indices = list(chain.from_iterable(output.argmax(dim=-1).tolist()))
        #print("3", flat_indices)

        # 將索引轉換為原始語言
        decoded_output = [reverse_output_vocab.get(idx, '<UNK>') for idx in flat_indices]
        #print("4", decoded_output)

        # 删除指定的單詞
        decoded_output = [word for word in decoded_output if word not in ['<PAD>', '<UNK>', '<BOS>', '<EOS>']]
        #print("5", decoded_output)

        # 合併處理後的單詞列表，以空格分隔
        merged_sentence = ' '.join(decoded_output)
        print("6", merged_sentence)

        # Compute the loss
        loss = criterion(output.view(-1, output_vocab_size), batch_output.view(-1))

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Print the loss
        print(f'Epoch {epoch+1}/{epochs}, Loss: {loss.item()}')

import csv

model.eval()

test_ids = []
predicted_outputs = []

with torch.no_grad():
    for test_input_sequence in test_input_tensor:
        test_input_sequence = test_input_sequence.unsqueeze(0).to(device)
        #print("0", batch_test_input)

        # 使用模型進行預測
        test_output = model(test_input_sequence, test_input_sequence)
        #print("1", test_output)

        # 建立反向的vocab
        reverse_output_vocab = {idx: word for word, idx in output_vocab.items()}
        #print("2", reverse_output_vocab)

        flat_indices = list(test_output.argmax(dim=-1).view(-1).tolist())
        #print("3", flat_indices)

        # 將索引轉換為原始語言
        decoded_output = [reverse_output_vocab.get(idx, '<UNK>') for idx in flat_indices]
        #print("4", decoded_output)

        # 删除指定的詞彙
        decoded_output = [word for word in decoded_output if word not in ['<PAD>', '<UNK>', '<BOS>', '<EOS>']]
        #print("5", decoded_output)

        # 合併處理後的詞彙列表，以空格分隔
        merged_sentence = ' '.join(decoded_output)
        #print("6", merged_sentence)

        predicted_outputs.append(merged_sentence)

for i, output in enumerate(predicted_outputs):
    print(f"Test Sample {i+1} Output: {output}")

# 要寫入的 CSV 文件名稱
output_csv_file = 'predicted_outputs.csv'

# 開啟 CSV 文件，並設定寫入模式
with open(output_csv_file, mode='w', newline='') as file:
    # 創建 CSV 寫入器
    writer = csv.writer(file)

    # 寫入 CSV 標頭
    writer.writerow(['id', 'txt'])

    # 寫入每個預測結果
    for i, output in enumerate(predicted_outputs):
        # 獲取對應的 test_id
        test_id = test_data_ZH['id'].values[i]

        # 寫入 id 和預測結果到 CSV 文件
        writer.writerow([test_id, output])

print(f"Predicted outputs saved to {output_csv_file}")

print("reverse_output_vocab:")
print(reverse_output_vocab)
