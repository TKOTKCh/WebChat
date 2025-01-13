from openai import OpenAI

from .retriever import ReferenceRetiever
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import re, os
from model.retriever.searching.bing_search import get_title_for_url
from model.utils import citation_correction
class WebChat:
    def __init__(self, openai_key, retriever_ckpt_path, device=None, filter_max_batch_size=400,
                 searcher_name="bing") -> None:
        self.device = device
        self.ref_retriever = ReferenceRetiever(retriever_ckpt_path, device, filter_max_batch_size, searcher_name)

        self.client = OpenAI(
            base_url="https://api.gpts.vin/v1",
            api_key=openai_key
        )

    def stream_query(self, question):
        print('start_query')
        refs = self.ref_retriever.query(question)
        if not refs:
            yield {"references": [], "answer": ""}
            return
        print('retriever query is done')
        for ref in refs:
            if len(ref['title']) == 0:
                ref['title'] = get_title_for_url(ref['url'])
        yield {"references": refs}
        prompt = ''
        for ix, ref in enumerate(refs):
            txt = ref["text"]
            prompt += f'参考 [{ix + 1}]: {txt}\n'
        prompt += f'请回答以下问题，注意需要以[x]标记出你引用参考资料的地方，比如[1]表示你引用了第一个参考资料，下面请回答:\n {question}'
        messages=[
            {"role": "system", "content": "请参考以下资料回答问题"},
            {"role": "user", "content": prompt}
        ]
        print(prompt)
        response = self.client.chat.completions.create(model="gpt-4-turbo", messages=messages)
        answer = response.choices[0].message.content
        answer = citation_correction(answer, [ref['text'] for ref in refs])
        yield {"answer": answer}



def load_webchat(args):
    # openai_key = args.webglm_ckpt_path or os.getenv("WEBGLM_CKPT") or 'THUDM/WebGLM'
    openai_key=os.getenv("USER_API_KEY")
    retiever_ckpt_path = os.getenv("WEBGLM_RETRIEVER_CKPT")
    if not retiever_ckpt_path:
        print(
            'Retriever checkpoint not specified, please specify it with --retriever_ckpt_path or $WEBGLM_RETRIEVER_CKPT')
        exit(1)

    print('WebChat Initializing...')

    webchat = WebChat(openai_key, retiever_ckpt_path, args.device, args.filter_max_batch_size, args.searcher)

    print('WebChat Loaded')

    return webchat