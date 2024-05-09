import os
import math
import jieba
import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.svm import SVC
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score

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
def ExtractParagraph(token_num,is_gram):
    cn_rawtxt_filePath = 'D:\invertiblity\programmes\PostGraduate\Python\Zipf_Law\cn_rawtxt'#获取中文语料库的文件夹
    cn_rawtxt_txtname = os.listdir(cn_rawtxt_filePath)#读取所有文件名
    txtnum = 0
    for txtname in cn_rawtxt_txtname:
        if txtname[-3:] == 'txt':
            txtnum = txtnum + 1
        else:
            cn_rawtxt_txtname.remove(txtname)
    train_gather = []
    label_gather = []
    for txtname in cn_rawtxt_txtname:#仅保留txt文件用于处理
        processed_cn_rawtxt_txtfilepath = cn_rawtxt_filePath + '\\' + txtname
        txtcontent = ReadTxtFile(processed_cn_rawtxt_txtfilepath)#读取原始文本
        processed_txtcontent_list = ProcessTxtContent(txtcontent)#按字分解，并去掉type = list
        seggenerator_txtcontent = jieba.cut("".join(processed_txtcontent_list))
        seglist_txtcontent = [segpart for segpart in seggenerator_txtcontent]#按词分解,type = list
        for idx_paragraph in range(math.ceil(1000 / txtnum)):
            if len(train_gather) < 1000:
                if is_gram:
                    single_paragraph_len = math.ceil(len(seglist_txtcontent) / math.ceil(1000 / txtnum))
                    train_gather.append(seglist_txtcontent[idx_paragraph * single_paragraph_len:idx_paragraph * single_paragraph_len + token_num])
                else:
                    single_paragraph_len = math.ceil(len(processed_txtcontent_list) / math.ceil(1000 / txtnum))
                    train_gather.append(processed_txtcontent_list[idx_paragraph * single_paragraph_len:idx_paragraph * single_paragraph_len + token_num])                    
                label_gather.append(txtname[:-4])
    return train_gather,label_gather[:1000]

if __name__ == '__main__':
    os.system('cls')
    token_num = 20
    topic_num = 50
    paragraph_num = 1000
    train_gather,label_gather = ExtractParagraph(token_num,False)#得到训练集与对应标签
    x_train, x_test, y_train, y_test = train_test_split(train_gather, label_gather, test_size=0.1, random_state=42)

    # 将文本转换为主题分布的流水线
    lda_pipeline = Pipeline([
        ('vectorizer', CountVectorizer(max_features=token_num, analyzer='char')),
        ('lda', LatentDirichletAllocation(n_components=topic_num, random_state=42, n_jobs=-1))
    ])
    # 将文本转换为主题分布
    x_train_lda = lda_pipeline.fit_transform([' '.join(x) for x in x_train])
    x_test_lda = lda_pipeline.transform([' '.join(x) for x in x_test])
    # 使用分类器进行训练和评估
    classifier = SVC()
    classifier.fit(x_train_lda, y_train)
    accuracy = np.mean(cross_val_score(classifier, x_train_lda, y_train, cv = 10))
    test_accuracy = accuracy_score(y_test, classifier.predict(x_test_lda))

    print("Token num = {},topic_num = {}".format(token_num,topic_num))
    print("Accuracy = {}%,Test_Accuracy = {}%".format(accuracy * 100,test_accuracy * 100))


