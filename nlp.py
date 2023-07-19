# -*- coding: utf-8 -*-


import torch
import torch.nn as nn
import random
import time
import numpy as np
import pkuseg
import argparse
from tqdm import trange
import os
from utils import read_corpus, batch_iter
from vocab import Vocab
from model import RNN_ATTs
import math
from sklearn.metrics import f1_score
import jieba

CUDA_LAUNCH_BLOCKING = 1

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    from tensorboardX import SummaryWriter

from transformers import AdamW, get_linear_schedule_with_warmup


def set_seed():
    random.seed(3344)
    np.random.seed(3344)
    torch.manual_seed(3344)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(3344)


def tokenizer(text):
    """
    定义TEXT的tokenize规则
    """
    # regex = re.compile(r'[^\u4e00-\u9fa5A-Za-z0-9]')
    # text = regex.sub(' ', text)
    seg = pkuseg.pkuseg()
    return [word for word in seg.cut(text) if word.strip()]


def train(args, model, train_data, dev_data, vocab, dtype='CNN'):
    LOG_FILE = args.output_file
    # 记录训练log
    with open(LOG_FILE, "a") as fout:
        fout.write('\n')
        fout.write('==========' * 6)
        fout.write('start trainning: {}'.format(dtype))
        fout.write('\n')

    time_start = time.time()
    if not os.path.exists(os.path.join('./runs', dtype)):
        os.makedirs(os.path.join('./runs', dtype))
    tb_writer = SummaryWriter(os.path.join('./runs', dtype))
    # 计算总的迭代次数
    t_total = args.num_epoch * (math.ceil(len(train_data) / args.batch_size))
    # optimizer = bnb.optim.Adam8bit(model.parameters(), lr=0.001, betas=(0.9, 0.995))  # add bnb optimizer
    optimizer = AdamW(model.parameters(), lr=args.learnning_rate, eps=1e-8)  # 设置优化器
    scheduler = get_linear_schedule_with_warmup(optimizer=optimizer, num_warmup_steps=args.warmup_steps,
                                                num_training_steps=t_total)  # 设置预热。
    criterion = nn.CrossEntropyLoss()  # 设置loss为交叉熵
    global_step = 0
    total_loss = 0.
    logg_loss = 0.
    val_acces = []
    train_epoch = trange(args.num_epoch, desc='train_epoch')
    for epoch in train_epoch:  # 训练epoch
        model.train()
        for src_sents, labels in batch_iter(train_data, args.batch_size, shuffle=True):
            src_sents = vocab.vocab.to_input_tensor(src_sents, args.device)
            global_step += 1
            optimizer.zero_grad()
            logits = model(src_sents)
            y_labels = torch.tensor(labels, device=args.device)
            example_losses = criterion(logits, y_labels)
            example_losses.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.GRAD_CLIP)
            optimizer.step()
            scheduler.step()

            total_loss += example_losses.item()
            if global_step % 100 == 0:
                loss_scalar = (total_loss - logg_loss) / 100
                logg_loss = total_loss

                with open(LOG_FILE, "a") as fout:
                    fout.write("epoch: {}, iter: {}, loss: {},learn_rate: {}\n".format(epoch, global_step, loss_scalar,
                                                                                       scheduler.get_lr()[0]))
                print("epoch: {}, iter: {}, loss: {}, learning_rate: {}".format(epoch, global_step, loss_scalar,
                                                                                scheduler.get_lr()[0]))
                tb_writer.add_scalar("lr", scheduler.get_lr()[0], global_step)
                tb_writer.add_scalar("loss", loss_scalar, global_step)

        print("Epoch", epoch, "Training loss", total_loss / global_step)
        eval_loss, eval_result = evaluate(args, criterion, model, dev_data, vocab)  # 评估模型
        with open(LOG_FILE, "a") as fout:
            fout.write("EVALUATE: epoch: {}, loss: {},eval_result: {}\n".format(epoch, eval_loss, eval_result))
        eval_acc = eval_result['acc']
        if len(val_acces) == 0 or eval_acc > max(val_acces):
            # 如果比之前的acc要da，就保存模型
            print("best model on epoch: {}, eval_acc: {}".format(epoch, eval_acc))
            torch.save(model.state_dict(), "classifa-best-{}.th".format(dtype))
            val_acces.append(eval_acc)

    time_end = time.time()
    print("run model of {},taking total {} m".format(dtype, (time_end - time_start) / 60))
    with open(LOG_FILE, "a") as fout:
        fout.write("run model of {},taking total {} m\n".format(dtype, (time_end - time_start) / 60))


def evaluate(args, criterion, model, dev_data, vocab):
    model.eval()
    total_loss = 0.
    total_step = 0.
    preds = None
    out_label_ids = None
    with torch.no_grad():  # 不需要更新模型，不需要梯度
        for src_sents, labels in batch_iter(dev_data, args.batch_size):
            # print(src_sents)
            src_sents = vocab.vocab.to_input_tensor(src_sents, args.device)
            logits = model(src_sents)
            # print(logits)
            labels = torch.tensor(labels, device=args.device)
            example_losses = criterion(logits, labels)

            total_loss += example_losses.item()
            total_step += 1

            if preds is None:
                preds = logits.detach().cpu().numpy()
                out_label_ids = labels.detach().cpu().numpy()
            else:
                preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
                out_label_ids = np.append(out_label_ids, labels.detach().cpu().numpy(), axis=0)

    preds = np.argmax(preds, axis=1)
    result = acc_and_f1(preds, out_label_ids)
    model.train()
    print("Evaluation loss", total_loss / total_step)
    print('Evaluation result', result)
    return total_loss / total_step, result


def acc_and_f1(preds, labels):
    acc = (preds == labels).mean()
    f1 = f1_score(y_true=labels, y_pred=preds, average='weighted')
    return {
        "acc": acc,
        "f1": f1,
        "acc_and_f1": (acc + f1) / 2,
    }


def build_vocab(args):
    if not os.path.exists(args.vocab_path):
        src_sents, labels = read_corpus(args.train_data_dir)
        labels = {label: idx for idx, label in enumerate(labels)}
        vocab = Vocab.build(src_sents, labels, args.max_vocab_size, args.min_freq)
        vocab.save(args.vocab_path)
    else:
        vocab = Vocab.load(args.vocab_path)
    return vocab


def main():
    torch.cuda.empty_cache()
    parse = argparse.ArgumentParser()

    parse.add_argument("--train_data_dir", default='./cnews/train.txt', type=str, required=False)
    parse.add_argument("--dev_data_dir", default='./cnews/val.txt', type=str, required=False)
    parse.add_argument("--test_data_dir", default='./cnews/test.txt', type=str, required=False)
    parse.add_argument("--output_file", default='deep_model.log', type=str, required=False)
    parse.add_argument("--batch_size", default=4, type=int)
    parse.add_argument("--do_train", default=True, action="store_true", help="Whether to run training.")
    parse.add_argument("--do_test", default=True, action="store_true", help="Whether to run training.")
    parse.add_argument("--learnning_rate", default=5e-4, type=float)
    parse.add_argument("--num_epoch", default=50, type=int)
    parse.add_argument("--max_vocab_size", default=50000, type=int)
    parse.add_argument("--min_freq", default=2, type=int)
    parse.add_argument("--hidden_size", default=256, type=int)
    parse.add_argument("--embed_size", default=300, type=int)
    parse.add_argument("--dropout_rate", default=0.2, type=float)
    parse.add_argument("--warmup_steps", default=0, type=int, help="Linear warmup over warmup_steps.")
    parse.add_argument("--GRAD_CLIP", default=1, type=float)
    parse.add_argument("--vocab_path", default='vocab.json', type=str)

    args = parse.parse_args()
    # 如果有GPU则使用gpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.device = device
    set_seed()
    # 缓存数据，再次使用增加速度。
    if os.path.exists('./cnews/cache_train_data'):
        train_data = torch.load('./cnews/cache_train_data')
    else:
        train_data = read_corpus(args.train_data_dir)
        train_data = [(text, labs) for text, labs in zip(*train_data)]
        torch.save(train_data, './cnews/cache_train_data')

    if os.path.exists('./cnews/cache_dev_data'):
        dev_data = torch.load('./cnews/cache_dev_data')
    else:
        dev_data = read_corpus(args.dev_data_dir)
        dev_data = [(text, labs) for text, labs in zip(*dev_data)]
        torch.save(dev_data, './cnews/cache_dev_data')

    vocab = build_vocab(args)
    label_map = vocab.labels

    # print(label_map)
    rnn_model = RNN_ATTs(len(vocab.vocab), args.embed_size, args.hidden_size,
                         len(label_map), n_layers=1, bidirectional=True, dropout=args.dropout_rate)
    rnn_model.load_state_dict(torch.load('classifa-best-RNN.th'))
    rnn_model.to(device)
    text = "不是年度最佳有点可惜。感觉要通关了，但是用了3个小时…… 从打巨大的boss开始，就觉得要通关了，但是用了近三个小时才通（避免剧透不多说了）。小岛这是要做电影吗？最后的这一段比一部电影也长呀，而且画面美轮美奂，不是年度最佳有点可惜了"
    predict(text, rnn_model, vocab, args)
    if args.do_train:
        rnn_model = RNN_ATTs(len(vocab.vocab), args.embed_size, args.hidden_size,
                             len(label_map), n_layers=1, bidirectional=True, dropout=args.dropout_rate)
        rnn_model.to(device)
        train(args, rnn_model, train_data, dev_data, vocab, dtype='RNN')

    if args.do_test:
        if os.path.exists('./cnews/cache_test_data'):
            test_data = torch.load('./cnews/cache_test_data')
            # print(test_data)
        else:
            test_data = read_corpus(args.test_data_dir)
            test_data = [(text, labs) for text, labs in zip(*test_data)]

            torch.save(test_data, './cnews/cache_test_data')

        cirtion = nn.CrossEntropyLoss()

        rnn_model = RNN_ATTs(len(vocab.vocab), args.embed_size, args.hidden_size,
                             len(label_map), n_layers=1, bidirectional=True, dropout=args.dropout_rate)
        rnn_model.load_state_dict(torch.load('classifa-best-RNN.th'))
        rnn_model.to(device)
        rnn_test_loss, rnn_result = evaluate(args, cirtion, rnn_model, test_data, vocab)
        with open(args.output_file, "a") as fout:
            fout.write('\n')
            fout.write('=============== test result ============\n')
            fout.write("test model of {}, loss: {},result: {}\n".format('RNN', rnn_test_loss, rnn_result))


# def predict(text, model, vocab, args):
#     model.eval()
#     with torch.no_grad():
#         # 对输入的文本进行分词
#         text = jieba.lcut(text)
#         print(text)
#         # 将文本转化为模型可以接受的形式
#         input_tensor = vocab.vocab.to_input_tensor([text], args.device)
#         print(input_tensor)
#         # 使用模型进行预测
#         logits = model(input_tensor)
#
#         # 选取预测结果中概率最大的类别
#         preds = torch.argmax(logits, dim=1)
#
#         # 将预测结果从tensor转化为numpy数组
#         preds = preds.detach().cpu().numpy()
#         print(preds[0])
#     model.train()
#     return preds[0]  # 如果只有一个句子，返回预测结果的第一个元素


def predict(text):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args = argparse.Namespace(
        vocab_path='vocab.json',
        hidden_size=256,
        dropout_rate=0.2,
        device=device
    )
    vocab = build_vocab(args)
    label_map = vocab.labels
    model = RNN_ATTs(len(vocab.vocab), 300, 256,
                     len(label_map), n_layers=1, bidirectional=True, dropout=args.dropout_rate)
    model.load_state_dict(torch.load('model.th'))
    model.to(device)
    model.eval()

    with torch.no_grad():
        # 对输入的文本进行分词
        text = jieba.lcut(text)
        # 将文本转化为模型可以接受的形式
        input_tensor = vocab.vocab.to_input_tensor([text], args.device)
        # 使用模型进行预测
        logits = model(input_tensor)
        # 选取预测结果中概率最大的类别
        preds = torch.argmax(logits, dim=1)
        # 将预测结果从tensor转化为numpy数组
        preds = preds.detach().cpu().numpy()

    return preds[0]  # 如果只有一个句子，返回预测结果的第一个元素



if __name__ == '__main__':
    parse = argparse.ArgumentParser()

    parse.add_argument("--train_data_dir", default='./cnews/train.txt', type=str, required=False)
    parse.add_argument("--dev_data_dir", default='./cnews/val.txt', type=str, required=False)
    parse.add_argument("--test_data_dir", default='./cnews/test.txt', type=str, required=False)
    parse.add_argument("--output_file", default='deep_model.log', type=str, required=False)
    parse.add_argument("--batch_size", default=4, type=int)
    parse.add_argument("--do_train", default=True, action="store_true", help="Whether to run training.")
    parse.add_argument("--do_test", default=True, action="store_true", help="Whether to run training.")
    parse.add_argument("--learnning_rate", default=5e-4, type=float)
    parse.add_argument("--num_epoch", default=50, type=int)
    parse.add_argument("--max_vocab_size", default=50000, type=int)
    parse.add_argument("--min_freq", default=2, type=int)
    parse.add_argument("--hidden_size", default=256, type=int)
    parse.add_argument("--embed_size", default=300, type=int)
    parse.add_argument("--dropout_rate", default=0.2, type=float)
    parse.add_argument("--warmup_steps", default=0, type=int, help="Linear warmup over warmup_steps.")
    parse.add_argument("--GRAD_CLIP", default=1, type=float)
    parse.add_argument("--vocab_path", default='vocab.json', type=str)

    args = parse.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.device = device
    vocab = build_vocab(args)
    label_map = vocab.labels
    rnn_model = RNN_ATTs(len(vocab.vocab), 300, 256,
                         len(label_map), n_layers=1, bidirectional=True, dropout=args.dropout_rate)
    rnn_model.load_state_dict(torch.load('model.th'))
    rnn_model.to(device)
    text = "油管霓虹妹妹的呐喊又让我忍不住回原神了 本帖最后由 风吹草动1979 于 2022-9-18 13:55 编辑昨天在B站看“巴黎剩母”的回归游戏视频，到晚上想看看她的直播，不小心看到原神3.1的前瞻。后来又在B站补了前边半小时没看的视频，意犹未尽，偶然听到霓虹妹妹的直播，我都惊了，真有这么夸张么！昨晚1点多才恋恋不舍的睡了，早上冷醒，想到可以试试领原神，就爬起来折腾吃灰几年的PS4，设置了很久的网络，终于可以下载原神了（PS4系统先升级到10.0），我的天，44GB，7个小时。真是不甘心，又去乱调网络的DNS，终于下降到4个小时。中午下载完，进游戏一看，显示4202代码。在网上一通搜索，原来要完全删掉重新下载才可以。那时的心情，就像霓虹妹妹已经满命的“七七”又抽到5星七七一样的心情吧。（https://www.bilibili.com/video/B ... aaf91e5c4194b049595）那就下吧，但不知怎么搞的，再次下载网络断了（可能是因为死机强制关机的缘故吧），wifi设置始终进不去，那就连网线吧，但需要这间屋网线的IP，要查IP又要把电脑搬来太麻烦了，偶然发现wifi又可以进去设置了。赶快设置好DNS下载，但下午的下载速率只有1M了，上午还是5M呢，又去换了DNS还是一样，算了，将就下吧，下完要十几个小时，可惜兑换码就过期了。后来下完17G的本体后在PS4安装后，也不能进入游戏。之后想到可以在电脑上兑换啊，可惜硬盘太小没空间了。今天上午又是5M速度，几下下完进游戏试着去兑换，果然过期了。后来又在网上看了许多的兑换码都不行，只有“yuanshen”这个才行（yuanshen2022都不行），也算是一种安慰吧。1.jpg(214.36 KB	 下载次数: 127)下载附件2022-9-18 13:54 上传2.jpg(291.49 KB	 下载次数: 132)下载附件2022-9-18 13:54 上传3.jpg(288.76 KB	 下载次数: 127)下载附件2022-9-18 13:55 上传"
    # predict(text, rnn_model, vocab, args)
    # text = "油管霓虹妹妹的呐喊又让我忍不住回原神了..."
    result = predict(text)
    print(result)
