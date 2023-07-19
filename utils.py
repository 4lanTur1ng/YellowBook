import pkuseg
import math
import random
from tqdm import tqdm,trange
import codecs

label_map =  {'DS': 0, 'SEKIRO': 1, 'genshin-impact': 2, 'MGS': 3, 'MH': 4, 'star-rail': 5}
def read_corpus(file_path):
    """读取语料
    :param file_path:
    :param type:
    :return:
    """
    src_data = []
    labels = []
    seg = pkuseg.pkuseg() #使用默认分词方式。
    with codecs.open(file_path,'r',encoding='utf-8') as fout:
        for line in tqdm(fout.readlines(),desc='reading corpus'):
            if line is not None:
                # line.strip()的意思是去掉每句话句首句尾的空格
                # .split(‘\t’)的意思是根据'\t'把label和文章内容分开，label和内容是通过‘\t’隔开的。
                # \t表示空四个字符，也称缩进，相当于按一下Tab键
                pair = line.strip().split('\t')
                if len(pair) != 2:
                    print(pair)
                    continue
                src_data.append(seg.cut(pair[1]))# 对文章内容分词。
                labels.append(pair[0])
    return (src_data, labels) #返回文章内容的分词结果和labels

def pad_sents(sents,pad_token):
    """pad句子"""
    sents_padded = []
    lengths = [len(s) for s in sents]
    max_len = max(lengths)
    for sent in sents:
        sent_padded = sent + [pad_token] * (max_len - len(sent))
        sents_padded.append(sent_padded)
    return sents_padded

def batch_iter(data, batch_size, shuffle=False):
    """
        batch数据
    :param data: list of tuple
    :param batch_size:
    :param shuffle:
    :return:
    """
    batch_num = math.ceil(len(data) / batch_size)# 计算迭代的次数
    index_array = list(range(len(data))) #按照data的长度，映射list
    if shuffle:#是否打乱顺序
        random.shuffle(index_array)

    for i in range(batch_num):
        indices = index_array[i*batch_size:(i+1)*batch_size]# 选出batchsize个index
        examples = [data[idx] for idx in indices]# 通过index找到对应的data
        examples = sorted(examples,key=lambda x: len(x[1]),reverse=True)#按照label排序
        src_sents = [e[0] for e in examples] #把data中的文章放到src_sents
        labels = [label_map[e[1]] for e in examples] #将标题映射label_map对应的value
        yield src_sents, labels
