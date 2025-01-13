
<h1>WebChat：基于网络检索增强的问答系统</h1>



## Overview
这是一个基于网络检索增强的问答系统WebChat，该系统由双阶段检索器和基于LLM的答案生成器构成，通过双阶段信息检索获取与用户问题相关的参考信息，去除网页中的无关上下文，提高模型生成效果。实验结果表明WebChat在WebCPM-QA和WebGLM-QA两个数据集上连续优于GPT系列基线，较GPT-4o-mini分别提升11.73\%和12.39\%。同时本文集成WebChat、WebCPM、WebGLM模型并利用Streanlit开发了一个网页原型系统。

## Requirements

要运行这个项目需要安装对应的依赖包，使用以下命令运行:

```
pip install -r requirements.txt
```

**NOTE**: 除依赖外，WebChat使用Contriever进行参考信息提取，这是一个无监督的预训练模型，需要从Huggingface中[facebook/contriever-msmarco · Hugging Face](https://huggingface.co/facebook/contriever-msmarco)下载对应模型权重至'mymodel'路径下。

此外原型系统也集成了WebCPM和WebGLM，如果您想要体验这两个模型的效果，需要下载对应模型权重[THUDM/WebGLM: WebGLM: An Efficient Web-enhanced Question Answering System (KDD 2023)](https://github.com/THUDM/WebGLM),[thunlp/WebCPM: Official codes for ACL 2023 paper "WebCPM: Interactive Web Search for Chinese Long-form Question Answering"](https://github.com/thunlp/WebCPM)

