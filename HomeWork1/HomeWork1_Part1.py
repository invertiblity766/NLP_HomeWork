import os
import jieba
import numpy as np
from matplotlib import pyplot as plt


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

if __name__ == '__main__':
    os.system('cls')
    cn_rawtxt_filePath = 'D:\invertiblity\programmes\PostGraduate\Python\Zipf_Law\cn_rawtxt'#获取中文语料库的文件夹
    cn_rawtxt_txtname = os.listdir(cn_rawtxt_filePath)#读取所有文件名
    for txtname in cn_rawtxt_txtname:#仅保留txt文件用于处理
        WordFrequency_Dict = {}#使用字典用于词频统计
        if txtname[-3:] == 'txt':
            processed_cn_rawtxt_txtfilepath = cn_rawtxt_filePath + '\\' + txtname
            txtcontent = ReadTxtFile(processed_cn_rawtxt_txtfilepath)#读取待处理的txt文件
            seggenerator_txtcontent = jieba.cut("".join(ProcessTxtContent(txtcontent)))#文本预处理
            seglist_txtcontent = [segpart for segpart in seggenerator_txtcontent]#得到分词数据
            for word in seglist_txtcontent:
                if word in WordFrequency_Dict.keys():
                    WordFrequency_Dict[word] = WordFrequency_Dict[word] + 1
                else:
                    WordFrequency_Dict[word] = 1
            sorted_WordFrequency_Dict = sorted(WordFrequency_Dict.items(),key= lambda kv:(kv[1],kv[0]),reverse= True)#得到降序排序的词频数据
            rank = [1]
            Current_rank = 1
            last_Frequency = sorted_WordFrequency_Dict[0][1]
            for i in range(1,len(sorted_WordFrequency_Dict)):
                Current_Frequency = sorted_WordFrequency_Dict[i][1]
                if Current_Frequency < last_Frequency:
                    Current_rank = Current_rank + 1
                rank.append(Current_rank)
                last_Frequency = Current_Frequency        
            Frequency = []
            for i in range(len(sorted_WordFrequency_Dict)):
                Frequency.append(sorted_WordFrequency_Dict[i][1])
            inv_rank = [1 / x for x in rank]
            plt.rcParams['font.sans-serif'] = ['SimHei']
            plt.plot(inv_rank,Frequency)
            plt.xlabel('inv_Rank')
            plt.ylabel('Frequecy')
            plt.title(txtname[:-4])
            plt.grid()
            plt.savefig('D:\invertiblity\programmes\PostGraduate\Python\Zipf_Law\cn_rawtxt\image'+ '\\' + txtname[:-4] + '.png')
            plt.close()