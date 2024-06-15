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

class RNNModel(tf.keras.Model):
    """docstring for RNNModel"""
    def __init__(self, hidden_size, hidden_layers, vocab_size):
        super(RNNModel, self).__init__()
        self.hidden_size = hidden_size
        self.hidden_layers = hidden_layers
        self.vocab_size = vocab_size

        self.embedding = tf.keras.layers.Embedding(vocab_size, hidden_layers)
        self.lstm_layers = [tf.keras.layers.LSTM(hidden_layers, return_sequences=True, return_state=True) for _ in
                            range(hidden_layers)]
        self.dense = tf.keras.layers.Dense(vocab_size)

    def call(self, inputs, states=None, return_state=False, training=False):
        x = self.embedding(inputs)
        new_states = []
        for i in range(self.hidden_layers):
            x, state_h, state_c = self.lstm_layers[i](x, initial_state=states[i] if states else None,
                                                      training=training)
            new_states.append([state_h, state_c])
        x = self.dense(x)
        if return_state:
            return x, new_states
        else:
            return x

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

    vocab_size = len(vocab)
    n_epoch = 10
    batch_size = 32
    time_steps = 15
    batch_nums = len(numdata) // (batch_size * time_steps)
    hidden_size = 256
    hidden_layers = 3
    learning_rate = 0.01

    model = RNNModel(hidden_size,hidden_layers,vocab_size)
    model.summary()
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate),loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True))
    train_data = Preprocess_agent.data_generator(numdata,batch_size,time_steps)
    model.fit(train_data,epochs = n_epoch,steps_per_epoch = batch_nums,callbacks=[history])
    model.save(r'seq2seq_model.h5')

    plt.plot(history.losses)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('seq2seq Training Loss')
    plt.show()
