import streamlit as st

from model.retriever.searching.bing_search import get_title_for_url
from mymodel import load_webchat

# from langchain_core.callbacks import CallbackManager

upload_dir = "datasets"

import os
os.environ['CUDA']='cuda:0'
os.environ['BING_SEARCH_KEY'] = '******'
os.environ['PYTHONPATH'] ='/mnt/d/Documents/VirtualMachineShare/Code/WebCPM-main/'
os.environ['WEBGLM_RETRIEVER_CKPT']='THUDM/webglm-contriever'
from openai import OpenAI

from model import citation_correction, load_model
import torch
from run_web_browsing.interaction import platformctrl_pipeline
import json
import re
import time
from cpm_live.generation.bee import CPMBeeBeamSearch
from cpm_live.models import CPMBeeTorch, CPMBeeConfig
from cpm_live.tokenizers import CPMBeeTokenizer
from run_web_browsing.utils import add_ref
import argparse
from arguments import add_model_config_args
import warnings
import json
warnings.filterwarnings("ignore")

_init_mes = ("你好！我是WebChat，一个基于网络增强的问答系统。作为你的智能伙伴，我支持联网搜索，能为你解决实时热点问题，请问有什么我能帮助您的?")
def clear_chat_history():
    st.session_state.messages = [{"role": "assistant", "content": _init_mes}]

ref_html = """

<details style="border: 1px solid #ccc; padding: 10px; border-radius: 4px; margin-bottom: 4px">
    <summary style="display: flex; align-items: center; font-weight: bold;">
        <span style="margin-right: 10px;">[{index}] {title}</span>
        <a href="{url}" style="text-decoration: none; background: none !important;" target="_blank">link</a>
    </summary>
    <p style="margin-top: 10px;">{text}</p>
</details>

"""

query_html="""
<details style="border: 1px solid #ccc; padding: 10px; border-radius: 4px; margin-bottom: 4px">
    <summary style="display: flex; align-items: center; font-weight: bold;">
        <span style="margin-right: 10px;"> {query}</span>
    </summary>
</details>
"""
def model_predict_cpmb(modelcpmb, tokenizer_cpmb, question, max_abstract_num=3, whether_corrupt_abstract=True):
    op = platformctrl_pipeline.Operator()
    beam_search = CPMBeeBeamSearch(
        model=modelcpmb,
        tokenizer=tokenizer_cpmb,
    )

    # topk_decoding = CPMBeeRandomSampling(
    #     model=modelcpmb,
    #     tokenizer=tokenizer_cpmb,
    # )

    # max length 4096, avoid memory overflow
    MEMORY_OVERFLOW_LENGTH = 3072

    def whether_filter_source(href, filter_hrefs):
        for k in filter_hrefs:
            if k in href:
                print("filtering the current source: " + href)
                return True
        return False

    with torch.inference_mode():
        question_input = question.strip()
        src_line = "原始问题：" + question_input + "搜索引擎查询语句："
        src_line = re.sub('<', '<<', src_line)
        instance = [
            {"source": src_line, "<ans>": ""},
        ]
        queries = beam_search.generate(instance, max_length=192, beam_size=3, repetition_penalty=1.05)[0][
            '<ans>'].split("；")
        queries = list(set(queries))
        print(queries)
        abstracts = []
        hrefs = []
        accessed_links = []
        # todo 不同query得到的data的重合网页需要去掉
        filter_hrefs = []
        # filter_hrefs = ['zhihu.com']
        for query in queries:
            print("current query is: " + query + "\n")
            query = re.sub(r'\n', '', query)
            searched_results = op.search(query)
            if len(searched_results) == 0:
                print("no results found in Bing for query: " + query)
                continue
            # visit 3 pages per query, you can increase the num
            max_page_per_query = min(3, op.get_page_num())
            for page_idx in range(max_page_per_query):
                if len(abstracts) >= max_abstract_num:
                    break
                if searched_results[page_idx]["url"] in accessed_links or whether_filter_source(
                        searched_results[page_idx]["url"], filter_hrefs):
                    continue
                else:
                    accessed_links.append(searched_results[page_idx]["url"])

                # src_line = "原始问题：" + question_input + "当前搜索引擎查询语句：" + query + "搜索引擎返回页面：" + "标题：" + searched_results[page_idx]["title"] + "；简介：" + searched_results[page_idx]["summary"] + "是否和原始问题相关？"
                # src_line = re.sub('<', '<<', src_line)
                # instance = [
                #     {"source": src_line, "<ans>": ""}
                # ]
                # whether_load_page = beam_search.generate(instance, max_length=8, beam_size=3, repetition_penalty=1.05)[0]["<ans>"]
                # if whether_load_page == "否":
                #     print("跳过页面：" + searched_results[page_idx]["title"] + "\n")
                #     continue
                # else:
                #     print("进入页面：" + searched_results[page_idx]["title"] + "\n")

                print("进入页面：" + searched_results[page_idx]["name"] + "\n")

                href, page_detail = op.load_page(page_idx)
                if page_detail is None:
                    print("your connection fails, no page rendered")
                    continue
                if len(page_detail) == 0:
                    print("page no content, continue \n")
                    continue
                print("页面内容：" + page_detail)
                print(href)
                print("\n")
                # 分window摘要
                extract_times = int(len(page_detail) / MEMORY_OVERFLOW_LENGTH) + 1
                abstract = []
                print("dividing into " + str(extract_times) + "sub pages")
                for idx in range(extract_times):
                    start = idx * MEMORY_OVERFLOW_LENGTH
                    end = (idx + 1) * MEMORY_OVERFLOW_LENGTH if (idx + 1) * MEMORY_OVERFLOW_LENGTH < len(
                        page_detail) else len(page_detail)
                    if end - start < 256:
                        print("remaining page too short, skip")
                        continue
                    detail = page_detail[start: end]
                    print("sub page " + str(idx) + ": \n" + detail)

                    src_line = "原始问题：" + question_input + "当前搜索引擎查询语句：" + query + "当前页面具体内容：" + detail + "摘取和原问题相关的摘要："
                    src_line = re.sub('<', '<<', src_line)
                    instance = [
                        {"source": src_line, "<ans>": ""}
                    ]
                    # sub_abstract = topk_decoding.generate(instance, max_length=512, top_p=0.9, repetition_penalty=1.05)[0]["<ans>"]
                    sub_abstract = \
                    beam_search.generate(instance, max_length=512, beam_size=3, repetition_penalty=1.05)[0]["<ans>"]
                    if sub_abstract != "无" and len(sub_abstract) != 0:
                        abstract.append(sub_abstract)
                        print("摘取内容：" + sub_abstract + "\n")
                    else:
                        print("无摘取 \n")
                if len(abstract) > 0:
                    hrefs.append(href)
                    abstracts.append("".join(abstract))
            if len(abstracts) >= max_abstract_num:
                break

        # print("loading QA model")
        # ckpt_path = "/data/private/qinyujia/CPM-Live-master/cpm-live/results/cpm_bee_qa_v2_new4-best.pt"
        # modelcpmb.load_state_dict(torch.load(ckpt_path))
        # modelcpmb.cuda()
        # print("loading ended")
        Qa_time = time.time()
        new_context = ""
        total_length = 0

        # if whether_corrupt_abstract:
        #     abstracts = currupt_abstract_v2(abstracts)

        for c in abstracts:
            if total_length >= MEMORY_OVERFLOW_LENGTH:
                break
            c_len = len(c)
            if total_length + c_len >= MEMORY_OVERFLOW_LENGTH:
                c = c[: MEMORY_OVERFLOW_LENGTH - total_length]
            total_length += c_len
            new_context += "摘要：" + c + "；"
        context = new_context
        src_line = context + "问题：" + question_input + "；答案："
        src_line = re.sub(' ', '', src_line)
        src_line = re.sub('<', '<<', src_line)
        instance = [
            {'source': src_line, '<ans>': ''},
        ]
        answer = beam_search.generate(instance, max_length=768, beam_size=3, repetition_penalty=1.05)[0]['<ans>']
        answer_addref = add_ref({'question': question, 'abstract': abstracts, 'href': hrefs, 'answer': answer})
        print('the question is:')
        print(question)
        print('the abstrcts are:')
        print(abstracts)
        print('the answer is:')
        print(answer)
        print('reference link')
        print(hrefs)
        print('addref answer')
        print(answer_addref)
        print('the time QA is:%f' % (time.time() - Qa_time))

        return {'question': question, 'abstract': abstracts, 'href': hrefs, 'answer': answer,
                'answer_add_ref': answer_addref}
# def load_webglm_model_sync(args):
#     # 创建并设置事件循环
#     loop = asyncio.new_event_loop()
#     asyncio.set_event_loop(loop)
#
#     # 异步执行加载模型
#     return loop.run_until_complete(load_model(args))
def main():

    st.set_page_config(page_title="WebChat: 基于网络检索增强的问答系统", page_icon="🤖", layout="wide")
    st.markdown(
        """
        <div style='text-align: center;'>
            <h1>WebChat</h1>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        """
        <div style='text-align: center;'>
            <h4>🤖 基于网络检索增强的问答系统 🤖</h4>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.sidebar:
        st.markdown("<h1 style='text-align:center;font-family:Georgia'>⚙️ WebChat </h1>", unsafe_allow_html=True)
        st.markdown("""信息检索课设：一个基于网络检索增强的问答系统\n""")

        st.markdown("-------")
        st.markdown("<h1 style='text-align:center;font-family:Georgia'>🌟Features</h1>", unsafe_allow_html=True)
        st.markdown(" - 🧾 集成WebGLM、WebCPM两个最新的网络检索增强型LLM")
        st.markdown(
            " - 🧾 支持网络搜索--通过LLM增强网络内容的检索能力,获得实时数据")
        st.markdown(
            " - 🧾 Web增强问答--利用LLM根据原问题与检索结果生成连贯问答")
        st.markdown("-------")
        st.markdown("<h1 style='text-align:center;font-family:Georgia'>🧾 How to use?</h1>",
                    unsafe_allow_html=True)
        st.markdown(
            "1. 选择模型，并输入对应API Key🔑")
        st.markdown(
            "2. 向模型提问💬")


        # 选择框的选项列表
        options = ['WebGLM', 'WebCPM','WebChat', 'gpt-3.5-turbo']
        # 在页面上显示选择框
        selected_model = st.selectbox(
            '请选择模型:',
            options,
        )

        if selected_model=='gpt-3.5-turbo' or selected_model=='WebChat':
            user_api_key = st.sidebar.text_input(
                label="#### 请输入openai API key👇", placeholder="Paste your openAI API key, sk-", type="password",
                key="openai_api_key"
            )
            if len(user_api_key)==0:
                st.stop()
            os.environ['USER_API_KEY'] = user_api_key
        else:

            bing_search_key = st.sidebar.text_input(
                label="#### 请输入Bing Search API key 👇", placeholder="Paste your Bing Search API key", type="password",
                key="bing_search_key"
            )
            if len(bing_search_key)==0:
                st.stop()

        if selected_model is not None:
            if selected_model=='gpt-3.5-turbo':
                if "client" not in st.session_state:
                    st.session_state.client = OpenAI(
                        base_url="https://api.gpts.vin/v1",
                        api_key="******"
                    )

            elif selected_model=='WebCPM':
                if 'webcpm' not in st.session_state and len(bing_search_key)==os.getenv("BING_SEARCH_KEY"):
                    config = CPMBeeConfig.from_json_file("./cpm_live/config/cpm-bee-10b.json")
                    if 'tokenizer_cpmb' not in st.session_state:
                        st.session_state.tokenizer_cpmb=CPMBeeTokenizer()
                    modelcpmb = CPMBeeTorch(config=config)
                    modelcpmb.load_state_dict(torch.load(r"./models/cpm_10b_webcpm_pipeline_finetuned.pt"))
                    modelcpmb.cuda()
                    if 'modelcpmb' not in st.session_state:
                        st.session_state.modelcpmb=modelcpmb

            elif selected_model=='WebGLM':
                if "webglm" not in st.session_state:
                    arg = argparse.ArgumentParser()
                    add_model_config_args(arg)
                    args = arg.parse_args()
                    st.session_state.webglm=load_model(args)
            else:
                if "webchat" not in st.session_state:
                    arg = argparse.ArgumentParser()
                    add_model_config_args(arg)
                    args = arg.parse_args()
                    st.session_state.webchat = load_webchat(args)
            st.write('你选择的是:', selected_model)
            st.sidebar.success(selected_model+" 加载成功", icon="🚀")
            st.sidebar.button('清除记录', on_click=clear_chat_history)
        st.markdown('-------')
        # st.markdown('Peking University')


    if "messages" not in st.session_state.keys():
        st.session_state.messages = [{"role": "assistant", "content": _init_mes}]


    # Display or clear chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])


    if question := st.chat_input():
        st.session_state.messages.append({"role": "user", "content": question})
        st.chat_message("user").write(question)
        if selected_model=='gpt-3.5-turbo':
            client=st.session_state.client
            response = client.chat.completions.create(model="gpt-3.5-turbo", messages=st.session_state.messages)
            msg = response.choices[0].message.content
            st.session_state.messages.append({"role": "assistant", "content": msg})
            st.chat_message("assistant").write(msg)
        elif selected_model=='WebGLM':
            webglm=st.session_state.webglm
            result=[]
            for resp in webglm.stream_query(question):
                if "references" in resp:
                    refs = resp["references"]
                    refs_html="<h4>参考链接</h4>" + "\n".join(
                        [ref_html.format(**item, index=idx + 1) for idx, item in enumerate(refs)])
                    st.markdown(refs_html,unsafe_allow_html=True)
                    # st.chat_message("assistant").write(refs)
                    result.append(refs)
                if "answer" in resp:
                    answer = resp["answer"]
                    answer = citation_correction(answer, [ref['text'] for ref in refs])
                    st.chat_message("assistant").write(answer)
                    result.append(answer)
            print(result)
            # print('here')
        elif selected_model == 'WebCPM':
            if 'modelcpmb' in st.session_state:
                modelcpmb=st.session_state.modelcpmb
                tokenizer_cpmb=st.session_state.tokenizer_cpmb
                pred_dict = model_predict_cpmb(modelcpmb, tokenizer_cpmb, question, max_abstract_num=5,
                                               whether_corrupt_abstract=False)
            else:
                with open(r'./run_web_browsing/predictions/test.json', 'r', encoding='utf-8',
                          errors='ignore') as file:
                    pred_dict = json.load(file)[0]
            hrefs=pred_dict['href']
            answer=pred_dict['answer_add_ref']
            abstracts=pred_dict['abstract']
            queries=pred_dict['queries']
            queries_str = '&nbsp;&nbsp;&nbsp;&nbsp;'.join(queries)

            refs=[]
            for index,url in enumerate(hrefs):
                title=get_title_for_url(url)
                refs.append({'url':url,'title':title,'text':abstracts[index]})
                if index==1:
                    message = "<h4>生成以下query</h4>\n" + query_html.format(query=queries_str)
                    st.markdown(message, unsafe_allow_html=True)
                    print(queries)
            refs_html = "<h4>参考链接</h4>" + "\n".join(
                [ref_html.format(**item, index=idx + 1) for idx, item in enumerate(refs)])

            st.markdown(refs_html, unsafe_allow_html=True)
            st.chat_message("assistant").write(answer)
            print(pred_dict)
        elif selected_model=='WebChat':
            webchat = st.session_state.webchat
            result = []
            for resp in webchat.stream_query(question):
                if "references" in resp:
                    refs = resp["references"]
                    refs_html = "<h4>参考链接</h4>" + "\n".join(
                        [ref_html.format(**item, index=idx + 1) for idx, item in enumerate(refs)])
                    st.markdown(refs_html, unsafe_allow_html=True)
                    # st.chat_message("assistant").write(refs)
                    result.append(refs)
                if "answer" in resp:
                    answer = resp["answer"]
                    st.chat_message("assistant").write(answer)
                    result.append(answer)
            print(result)

if __name__ == "__main__":
    main()
