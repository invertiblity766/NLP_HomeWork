<<<<<<< HEAD
<<<<<<< HEAD
import os
import math
import jieba
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
def get_UniWord_Fre(Wordlist):
    Word_Fre = {}
    for word in Wordlist:
        if word in Word_Fre.keys():
            Word_Fre[word] = Word_Fre[word] + 1
        else:
            Word_Fre[word] = 1
    return Word_Fre
def Cal_Uniword_Entropy(Wordlist,is_gram):
    Word_Fre = get_UniWord_Fre(Wordlist)
    TotFre = sum([item[1] for item in Word_Fre.items()])
    Uniword_Entropy = sum([-(Fre / TotFre) * math.log((Fre / TotFre),2) for Fre in Word_Fre.values()])
    if is_gram:
        print("基于词的一阶模型的中文信息熵为:{}比特/词".format(Uniword_Entropy))
    else:
        print("基于字的一阶模型的中文信息熵为:{}比特/字".format(Uniword_Entropy))
def get_BinWord_Fre(wordlist):
    Word_Fre = {}
    for i in range(len(wordlist) - 1):
        if (wordlist[i],wordlist[i + 1]) in Word_Fre.keys():
            Word_Fre[(wordlist[i],wordlist[i + 1])] = Word_Fre[(wordlist[i],wordlist[i + 1])] + 1
        else:
            Word_Fre[(wordlist[i],wordlist[i + 1])] = 1
    return Word_Fre
def Cal_BinWord_Entropy(Wordlist,is_gram):
    UniWord_Fre = get_UniWord_Fre(Wordlist)
    BinWord_Fre = get_BinWord_Fre(Wordlist)
    TotFre_Bin = sum([item[1] for item in BinWord_Fre.items()])
    Binword_Entropy = 0.0
    for word in BinWord_Fre.keys():
        BinWord_absPro = BinWord_Fre[word] / TotFre_Bin
        BinWord_revPro = BinWord_Fre[word] / UniWord_Fre[word[0]]
        Binword_Entropy = Binword_Entropy - BinWord_absPro * math.log(BinWord_revPro,2)
    if is_gram:
        print("基于词的二阶模型的中文信息熵为:{}比特/词".format(Binword_Entropy))
    else:
        print("基于字的二阶模型的中文信息熵为:{}比特/字".format(Binword_Entropy))
if __name__ == '__main__':
    os.system('cls')
    cn_rawtxt_filePath = 'D:\invertiblity\programmes\PostGraduate\Python\Zipf_Law\cn_rawtxt'#获取中文语料库的文件夹
    cn_rawtxt_txtname = os.listdir(cn_rawtxt_filePath)#读取所有文件名
    for txtname in cn_rawtxt_txtname:#仅保留txt文件用于处理
        if txtname[-3:] == 'txt':
            processed_cn_rawtxt_txtfilepath = cn_rawtxt_filePath + '\\' + txtname
            txtcontent = ReadTxtFile(processed_cn_rawtxt_txtfilepath)#读取原始文本
            processed_txtcontent_list = ProcessTxtContent(txtcontent)#按字分解，并去掉type = list
            seggenerator_txtcontent = jieba.cut("".join(processed_txtcontent_list))
            seglist_txtcontent = [segpart for segpart in seggenerator_txtcontent]#按词分解,type = list
            print("--------------------" + txtname + "--------------------")
            Cal_Uniword_Entropy(seglist_txtcontent,True)
            Cal_Uniword_Entropy(processed_txtcontent_list,False)
            Cal_BinWord_Entropy(seglist_txtcontent,True)
            Cal_BinWord_Entropy(processed_txtcontent_list,False)
    
    
=======
import os
import math
import jieba
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
def get_UniWord_Fre(Wordlist):
    Word_Fre = {}
    for word in Wordlist:
        if word in Word_Fre.keys():
            Word_Fre[word] = Word_Fre[word] + 1
        else:
            Word_Fre[word] = 1
    return Word_Fre
def Cal_Uniword_Entropy(Wordlist,is_gram):
    Word_Fre = get_UniWord_Fre(Wordlist)
    TotFre = sum([item[1] for item in Word_Fre.items()])
    Uniword_Entropy = sum([-(Fre / TotFre) * math.log((Fre / TotFre),2) for Fre in Word_Fre.values()])
    if is_gram:
        print("基于词的一阶模型的中文信息熵为:{}比特/词".format(Uniword_Entropy))
    else:
        print("基于字的一阶模型的中文信息熵为:{}比特/字".format(Uniword_Entropy))
def get_BinWord_Fre(wordlist):
    Word_Fre = {}
    for i in range(len(wordlist) - 1):
        if (wordlist[i],wordlist[i + 1]) in Word_Fre.keys():
            Word_Fre[(wordlist[i],wordlist[i + 1])] = Word_Fre[(wordlist[i],wordlist[i + 1])] + 1
        else:
            Word_Fre[(wordlist[i],wordlist[i + 1])] = 1
    return Word_Fre
def Cal_BinWord_Entropy(Wordlist,is_gram):
    UniWord_Fre = get_UniWord_Fre(Wordlist)
    BinWord_Fre = get_BinWord_Fre(Wordlist)
    TotFre_Bin = sum([item[1] for item in BinWord_Fre.items()])
    Binword_Entropy = 0.0
    for word in BinWord_Fre.keys():
        BinWord_absPro = BinWord_Fre[word] / TotFre_Bin
        BinWord_revPro = BinWord_Fre[word] / UniWord_Fre[word[0]]
        Binword_Entropy = Binword_Entropy - BinWord_absPro * math.log(BinWord_revPro,2)
    if is_gram:
        print("基于词的二阶模型的中文信息熵为:{}比特/词".format(Binword_Entropy))
    else:
        print("基于字的二阶模型的中文信息熵为:{}比特/字".format(Binword_Entropy))
if __name__ == '__main__':
    os.system('cls')
    cn_rawtxt_filePath = 'D:\invertiblity\programmes\PostGraduate\Python\Zipf_Law\cn_rawtxt'#获取中文语料库的文件夹
    cn_rawtxt_txtname = os.listdir(cn_rawtxt_filePath)#读取所有文件名
    for txtname in cn_rawtxt_txtname:#仅保留txt文件用于处理
        if txtname[-3:] == 'txt':
            processed_cn_rawtxt_txtfilepath = cn_rawtxt_filePath + '\\' + txtname
            txtcontent = ReadTxtFile(processed_cn_rawtxt_txtfilepath)#读取原始文本
            processed_txtcontent_list = ProcessTxtContent(txtcontent)#按字分解，并去掉type = list
            seggenerator_txtcontent = jieba.cut("".join(processed_txtcontent_list))
            seglist_txtcontent = [segpart for segpart in seggenerator_txtcontent]#按词分解,type = list
            print("--------------------" + txtname + "--------------------")
            Cal_Uniword_Entropy(seglist_txtcontent,True)
            Cal_Uniword_Entropy(processed_txtcontent_list,False)
            Cal_BinWord_Entropy(seglist_txtcontent,True)
            Cal_BinWord_Entropy(processed_txtcontent_list,False)
    
    
>>>>>>> b0de3182a8f8df1e001d268dd1503bcd113ee125
=======
import os
import math
import jieba
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
def get_UniWord_Fre(Wordlist):
    Word_Fre = {}
    for word in Wordlist:
        if word in Word_Fre.keys():
            Word_Fre[word] = Word_Fre[word] + 1
        else:
            Word_Fre[word] = 1
    return Word_Fre
def Cal_Uniword_Entropy(Wordlist,is_gram):
    Word_Fre = get_UniWord_Fre(Wordlist)
    TotFre = sum([item[1] for item in Word_Fre.items()])
    Uniword_Entropy = sum([-(Fre / TotFre) * math.log((Fre / TotFre),2) for Fre in Word_Fre.values()])
    if is_gram:
        print("基于词的一阶模型的中文信息熵为:{}比特/词".format(Uniword_Entropy))
    else:
        print("基于字的一阶模型的中文信息熵为:{}比特/字".format(Uniword_Entropy))
def get_BinWord_Fre(wordlist):
    Word_Fre = {}
    for i in range(len(wordlist) - 1):
        if (wordlist[i],wordlist[i + 1]) in Word_Fre.keys():
            Word_Fre[(wordlist[i],wordlist[i + 1])] = Word_Fre[(wordlist[i],wordlist[i + 1])] + 1
        else:
            Word_Fre[(wordlist[i],wordlist[i + 1])] = 1
    return Word_Fre
def Cal_BinWord_Entropy(Wordlist,is_gram):
    UniWord_Fre = get_UniWord_Fre(Wordlist)
    BinWord_Fre = get_BinWord_Fre(Wordlist)
    TotFre_Bin = sum([item[1] for item in BinWord_Fre.items()])
    Binword_Entropy = 0.0
    for word in BinWord_Fre.keys():
        BinWord_absPro = BinWord_Fre[word] / TotFre_Bin
        BinWord_revPro = BinWord_Fre[word] / UniWord_Fre[word[0]]
        Binword_Entropy = Binword_Entropy - BinWord_absPro * math.log(BinWord_revPro,2)
    if is_gram:
        print("基于词的二阶模型的中文信息熵为:{}比特/词".format(Binword_Entropy))
    else:
        print("基于字的二阶模型的中文信息熵为:{}比特/字".format(Binword_Entropy))
if __name__ == '__main__':
    os.system('cls')
    cn_rawtxt_filePath = 'D:\invertiblity\programmes\PostGraduate\Python\Zipf_Law\cn_rawtxt'#获取中文语料库的文件夹
    cn_rawtxt_txtname = os.listdir(cn_rawtxt_filePath)#读取所有文件名
    for txtname in cn_rawtxt_txtname:#仅保留txt文件用于处理
        if txtname[-3:] == 'txt':
            processed_cn_rawtxt_txtfilepath = cn_rawtxt_filePath + '\\' + txtname
            txtcontent = ReadTxtFile(processed_cn_rawtxt_txtfilepath)#读取原始文本
            processed_txtcontent_list = ProcessTxtContent(txtcontent)#按字分解，并去掉type = list
            seggenerator_txtcontent = jieba.cut("".join(processed_txtcontent_list))
            seglist_txtcontent = [segpart for segpart in seggenerator_txtcontent]#按词分解,type = list
            print("--------------------" + txtname + "--------------------")
            Cal_Uniword_Entropy(seglist_txtcontent,True)
            Cal_Uniword_Entropy(processed_txtcontent_list,False)
            Cal_BinWord_Entropy(seglist_txtcontent,True)
            Cal_BinWord_Entropy(processed_txtcontent_list,False)
    
    
>>>>>>> b0de318 (第一次作业——20240410)
