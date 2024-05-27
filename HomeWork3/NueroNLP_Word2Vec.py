import os
import jieba
import gensim
import numpy as np
from matplotlib import pyplot as plt
from gensim.models import Word2Vec

def ReadTxtFile(TxtFilepath):
    with open(TxtFilepath,"r",encoding="gbk",errors='ignore') as data:
        txtcontent = data.read()
        return txtcontent
def is_chinese(ch):
    if '\u4e00'<= ch <= '\u9fff':
        return True
    return False
def ProcessTxtContent(TxtContent):
    processed_txtContent = []
    stopwords = ReadTxtFile('cn_stopwords.txt')
    for ch in TxtContent:
        if is_chinese(ch) and (ch not in stopwords):
            processed_txtContent.append(ch)
    return processed_txtContent
def get_txtname():
    cn_rawtxt_filePath = 'D:\invertiblity\programmes\PostGraduate\Python\Zipf_Law\cn_rawtxt'#获取中文语料库的文件夹
    cn_rawtxt_txtname = os.listdir(cn_rawtxt_filePath)#读取所有文件名
    for txtname in cn_rawtxt_txtname:
        if txtname[-3:] != 'txt':
            cn_rawtxt_txtname.remove(txtname)
    return cn_rawtxt_txtname
def ExtractSource(is_gram):
    train_gather = []
    cn_rawtxt_txtname = get_txtname()#读取所有文件名
    cn_rawtxt_filePath = 'D:\invertiblity\programmes\PostGraduate\Python\Zipf_Law\cn_rawtxt'#获取中文语料库的文件夹
    for txtname in cn_rawtxt_txtname[:1]:#仅保留txt文件用于处理
        processed_cn_rawtxt_txtfilepath = cn_rawtxt_filePath + '\\' + txtname
        txtcontent = ReadTxtFile(processed_cn_rawtxt_txtfilepath)#读取原始文本
        processed_txtcontent_list = ProcessTxtContent(txtcontent)#按字分解，并去掉type = list
        seggenerator_txtcontent = jieba.cut("".join(processed_txtcontent_list))
        seglist_txtcontent = [segpart for segpart in seggenerator_txtcontent]#按词分解,type = list
        if is_gram:
            train_gather.append(seglist_txtcontent)
        else:
            train_gather.append(processed_txtcontent_list)
    return train_gather



if __name__ == '__main__':
    os.system('cls')
    gram_source = ExtractSource(True)
    Average_cosine_similary = []
    n_epoch = 20
    for i_epoch in range(1,n_epoch + 1):
        model = Word2Vec(sentences=gram_source,epochs = i_epoch,min_count= 3,vector_size= 100,window = 1)
        Average_cosine_similary.append(np.mean([model.wv.most_similar(word,topn = 1)[0][1] for word in model.wv.index_to_key]))
        print("Average Cosine similarity after epoch {}:{}".format(i_epoch,Average_cosine_similary[-1]))
    plt.plot(range(1,n_epoch + 1),Average_cosine_similary)
    plt.axis([1,n_epoch,0.0,1.0])
    plt.xlabel('epoch')
    plt.ylabel('Average cosine similarity')
    plt.grid()
    plt.show()

