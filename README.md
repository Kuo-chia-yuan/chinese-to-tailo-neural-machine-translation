# chinese-to-tailo-neural-machine-translation

## 介紹
將中文翻譯成閩南語
 - input：中文句子
 - output：閩南語句子

## transformer 架構 & 作法說明
```
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
```

1. Input Embedding: 將 input 序列中的每個元素映射到一個高維度空間中的向量。
2. Positional Encoding: 為了使模型能夠處理序列的 data，需要將序列中元素的相對位置資訊加入到模型中。
3. Self-Attention Mechanism: 為 transformer 關鍵的一步，此時會讓模型在處理序列時，替每個位置分配不同的權重，而不僅僅是固定的權重，如此一來，模型即可更好的處理不同位置之間的關聯性。
4. Multi-Head Attention: 為了提高模型對不同表示空間的建模能力，因此引入多頭的自注意力機制，每個 head 分別學習不同的表示。
5. Feedforward Network: 在每個位置上加上一個全連接層的前饋網路，增強模型的非線性建模能力。
6. Residual Connection and Layer Normalization: 每個子層 (注意力 + 前饋網路) 後都有一個殘差連接層和歸一化層，有助於減緩訓練中的梯度消失問題。
7. Encoder-decoder structure: 編碼器處理 input 序列，解碼器生成 output 序列。
8. Output Layer: 模型最後會通過一個全連接層輸出最終的預測結果。

## 其他 function & 參數
  - loss: CrossEntropyLoss
  - optimizer: Adam
  - avtivation: Relu
  - learning rate = 0.001
  - batch_size = 64
  - d_model = 256
  - n_head = 4
  - num_encoder_layers = 3
  - num_decoder_layers = 3
  - dim_feedforward = 2048
  - dropout = 0.1

## 訓練結果
  - 最終的 loss 降至 0.05 左右
  
    ![image](https://github.com/Kuo-chia-yuan/chinese-to-tailo-neural-machine-translation/assets/56677419/d8a9a60a-d417-4817-84b4-896265455762)

## 遇到問題 & 解決方法
  1. input_data 和 output_data 的前處理較複雜，一開始我將中文和台語都一個字一個字切割，結果發現這樣會導致單字之間的連貫性被破壞，詞彙無法一一對應，所以後來改用一個詞彙一個詞彙切割，兩者才能對應到正確的詞彙。
     ```
     data_ZH.rename(columns={"id": "id", "txt": "input"}, inplace=True)
     data_TL.rename(columns={"id": "id", "txt": "output"}, inplace=True)
     data_ZH['input'] = data_ZH['input'].apply(lambda x: x.split() if isinstance(x, str) else [])
     data_TL['output'] = data_TL['output'].apply(lambda x: x.split() if isinstance(x, str) else [])
     ```
     ![image](https://github.com/Kuo-chia-yuan/chinese-to-tailo-neural-machine-translation/assets/56677419/ebd89c5a-83db-40a5-8480-faa861e3b991)

  2. 因為有太多中文和台語單詞，所以一開始不知道要怎麼創建詞彙表 (vocab)，後來透過依照 id 順序搜索每句話的單詞，並給予編號，若先前已出現過的話，則不予新的編號。input_vocab 和 output_vocab 都完成後，需要將所有單詞轉成數字，形成數字序列 (input_num 和 output_num)，以利後續的 training。
     ```
     tokenized_data = data_ZH['input'].tolist()
     input_vocab = {'<PAD>': 0, '<UNK>': 1, '<BOS>': 2, '<EOS>': 3}
      
     for sentence in tokenized_data:
          for word in sentence:
              if word not in input_vocab:
                  input_vocab[word] = len(input_vocab)
      
     input_num = [[input_vocab.get(word, input_vocab['<UNK>']) for word in sentence] for sentence in tokenized_data]
     ```
     ![image](https://github.com/Kuo-chia-yuan/chinese-to-tailo-neural-machine-translation/assets/56677419/3072a5b7-3187-4ccf-9dc3-34b617e94164)

  3. 當輸出 test_input 的預測結果後，會得到 641 個數字序列，此時需要先求出 reverse_output_vocab，才能將數字序列轉回台語。得到 reverse_output_vocab 後，還要刪除句子中的 'PAD', 'UNK', 'BOS', 'EOS' 四個詞彙，並將同一句的台語單詞合併，且用空格分隔。
     ```
     # 建立反向的vocab
     reverse_output_vocab = {idx: word for word, idx in output_vocab.items()}
     flat_indices = list(test_output.argmax(dim=-1).view(-1).tolist())
     # 將索引轉換為原始語言
     decoded_output = [reverse_output_vocab.get(idx, '<UNK>') for idx in flat_indices]
     # 删除指定的詞彙
     decoded_output = [word for word in decoded_output if word not in ['<PAD>', '<UNK>', '<BOS>', '<EOS>']]
     # 合併處理後的詞彙列表，以空格分隔
     merged_sentence = ' '.join(decoded_output)
     ```
     ![image](https://github.com/Kuo-chia-yuan/chinese-to-tailo-neural-machine-translation/assets/56677419/16db2620-b063-4e21-84ed-200ea84058f9)
