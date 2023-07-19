import torch
import torch.nn.functional as F

# 加载模型
model = torch.jit.load('model.th')


# 创建预处理函数
def preprocess(text):
    # 在这里进行文本预处理，如分词、向量化等
    # 返回处理后的文本数据
    processed_text = text.lower()  # 示例：将文本转换为小写
    return processed_text


# 创建预测函数
def predict_partition(text):
    # 对文本进行预处理
    processed_text = preprocess(text)

    # 将文本转换为模型所需的输入张量
    input_tensor = torch.tensor(processed_text)  # 示例：将文本转换为张量

    # 将输入张量传递给模型进行预测
    output = model(input_tensor)

    # 对输出进行后处理，如取最大值索引等
    predicted_partition = torch.argmax(output)

    return predicted_partition


print(predict_partition("I love you"))


