import os
import jieba
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


class preprocess():
    def __init__(self,cn_rawtxt_filepath):
        self.cn_rawtxt_filepath = cn_rawtxt_filepath
    def is_chinese(self,ch):
        if '\u4e00'<= ch <= '\u9fff':
            return True
        return False
    def ReadTxtFile(self,TxtFilepath):
        with open(TxtFilepath,"r",encoding="gbk",errors='ignore') as data:
            txtcontent = data.read()
            return txtcontent
    def get_txtname(self):
        cn_rawtxt_filePath = self.cn_rawtxt_filepath#获取中文语料库的文件夹
        cn_rawtxt_txtname = os.listdir(cn_rawtxt_filePath)#读取所有文件名
        for txtname in cn_rawtxt_txtname:
            if txtname[-3:] != 'txt':
                cn_rawtxt_txtname.remove(txtname)
        return cn_rawtxt_txtname
    def ProcessTxtContent(self,TxtContent):
        processed_txtContent = []
        stopwords = self.ReadTxtFile('cn_stopwords.txt')
        for ch in TxtContent:
            if self.is_chinese(ch) and (ch not in stopwords):
                processed_txtContent.append(ch)
        return processed_txtContent
    def ExtractSource(self):
        Source = []
        cn_rawtxt_txtname = self.get_txtname()
        cn_rawtxt_filePath = self.cn_rawtxt_filepath
        for txtname in cn_rawtxt_txtname[:1]:#仅保留txt文件用于处理
            processed_cn_rawtxt_txtfilepath = cn_rawtxt_filePath + '\\' + txtname
            txtcontent = self.ReadTxtFile(processed_cn_rawtxt_txtfilepath)#读取原始文本
            processed_txtcontent_list = self.ProcessTxtContent(txtcontent)#按字分解，并去掉type = list
            seggenerator_txtcontent = jieba.cut("".join(processed_txtcontent_list))
            seglist_txtcontent = [segpart for segpart in seggenerator_txtcontent]#按词分解,type = list
            Source = Source + seglist_txtcontent
        return Source
    def data_generator(self,data,batch_size,time_steps):
        data = data[:(len(data) // (batch_size * time_steps)) * batch_size * time_steps]
        data = np.array(data).reshape((batch_size,-1))
        while True:
            for i in range(0,data.shape[1],time_steps):
                x = data[:,i:i + time_steps]
                y = np.roll(x,-1,axis = 1)
                yield x,y

# ========Transformer编码器========
class TransformerEncoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dff, rate=0.1):
        super(TransformerEncoderLayer, self).__init__()
        self.mha = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=d_model)
        self.ffn = tf.keras.Sequential([
            tf.keras.layers.Dense(dff, activation='relu'),
            tf.keras.layers.Dense(d_model)
        ])
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)

    def call(self, x, training, mask):
        attn_output = self.mha(x, x, x, attention_mask=mask)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(x + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)

# ========Transformer解码器========
class TransformerDecoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dff, rate=0.1):
        super(TransformerDecoderLayer, self).__init__()
        self.mha1 = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=d_model)
        self.mha2 = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=d_model)
        self.ffn = tf.keras.Sequential([
            tf.keras.layers.Dense(dff, activation='relu'),
            tf.keras.layers.Dense(d_model)
        ])
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)
        self.dropout3 = tf.keras.layers.Dropout(rate)

    def call(self, x, enc_output, training, look_ahead_mask, padding_mask):
        attn1 = self.mha1(x, x, x, attention_mask=look_ahead_mask)
        attn1 = self.dropout1(attn1, training=training)
        out1 = self.layernorm1(x + attn1)
        attn2 = self.mha2(out1, enc_output, enc_output, attention_mask=padding_mask)
        attn2 = self.dropout2(attn2, training=training)
        out2 = self.layernorm2(out1 + attn2)
        ffn_output = self.ffn(out2)
        ffn_output = self.dropout3(ffn_output, training=training)
        return self.layernorm3(out2 + ffn_output)

# ========完整的Transformer模型========
class TransformerModel(tf.keras.Model):
    def __init__(self, num_layers, d_model, num_heads, dff, input_vocab_size, target_vocab_size, pe_input, pe_target, rate=0.1):
        super(TransformerModel, self).__init__()
        self.encoder_embedding = tf.keras.layers.Embedding(input_vocab_size, d_model)
        self.decoder_embedding = tf.keras.layers.Embedding(target_vocab_size, d_model)
        self.encoder_layers = [TransformerEncoderLayer(d_model, num_heads, dff, rate) for _ in range(num_layers)]
        self.decoder_layers = [TransformerDecoderLayer(d_model, num_heads, dff, rate) for _ in range(num_layers)]
        self.final_layer = tf.keras.layers.Dense(target_vocab_size)

    def call(self, inputs, training=False):
        inp, tar = inputs
        enc_padding_mask, look_ahead_mask, dec_padding_mask = self.create_masks(inp, tar)
        enc_output = self.encoder_embedding(inp)
        for i in range(len(self.encoder_layers)):
            enc_output = self.encoder_layers[i](enc_output, training, enc_padding_mask)
        dec_output = self.decoder_embedding(tar)
        for i in range(len(self.decoder_layers)):
            dec_output = self.decoder_layers[i](dec_output, enc_output, training, look_ahead_mask, dec_padding_mask)
        final_output = self.final_layer(dec_output)
        return final_output

    def create_masks(self, inp, tar):
        enc_padding_mask = self.create_padding_mask(inp)
        dec_padding_mask = self.create_padding_mask(inp)
        look_ahead_mask = self.create_look_ahead_mask(tf.shape(tar)[1])
        dec_target_padding_mask = self.create_padding_mask(tar)
        combined_mask = tf.maximum(dec_target_padding_mask, look_ahead_mask)
        return enc_padding_mask, combined_mask, dec_padding_mask

    def create_padding_mask(self, seq):
        seq = tf.cast(tf.math.equal(seq, 0), tf.float32)
        return seq[:, tf.newaxis, tf.newaxis, :]

    def create_look_ahead_mask(self, size):
        mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
        return mask

# ========定义回调函数========
class LossHistory(tf.keras.callbacks.Callback):
    def on_train_begin(self, logs=None):
        self.losses = []

    def on_epoch_end(self, epoch, logs=None):
        self.losses.append(logs.get('loss'))

history = LossHistory()


if __name__ == '__main__':
    os.system('cls')

    Preprocess_agent = preprocess('D:\invertiblity\programmes\PostGraduate\Python\Zipf_Law\cn_rawtxt')
    source_data = Preprocess_agent.ExtractSource()
    vocab = list(set(source_data))
    word2id = {c:i for i,c in enumerate(vocab)}
    id2word = {i:c for i,c in enumerate(vocab)}
    numdata = [word2id[word] for word in source_data]

    num_layers = 4
    d_model = 128
    dff = 512
    num_heads = 8
    input_vocab_size = len(vocab) + 2
    target_vocab_size = len(vocab) + 2
    dropout_rate = 0.1

    batch_size = 32
    time_steps = 60
    epochs = 100
    learning_rate = 0.001

    model = TransformerModel(num_layers, d_model, num_heads, dff, input_vocab_size, target_vocab_size, time_steps, time_steps, dropout_rate)

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate),
                loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True))

    train_data = Preprocess_agent.data_generator(numdata, batch_size, time_steps)

    model.fit(train_data, epochs=epochs, steps_per_epoch=len(numdata) // (batch_size * time_steps), callbacks=[history])

    # =======绘制Loss曲线========
    plt.plot(history.losses)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('transformer Training Loss')
    plt.show()