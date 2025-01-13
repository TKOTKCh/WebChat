from openai import OpenAI
import json
from sklearn.metrics import f1_score, precision_score, recall_score
from rouge import Rouge
import numpy as np
import jieba
import string


def get_ngram(text, n):
    ngram_list = []
    for i in range(len(text) - n + 1):
        ngram_list.append(text[i: i + n])
    return list(set(ngram_list))


def get_novelty(source, target, n):
    ngram_source = get_ngram(source, n)
    ngram_target = get_ngram(target, n)
    return float(len([x for x in ngram_target if x not in ngram_source])) / len(ngram_target)

def get_rouge_over_list(prediction, groundtruth):
    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)
    if len(remove_punc(prediction)) == 0:
        return 0.0
    rouge = Rouge()
    if type(groundtruth)==list:
        if len(groundtruth)==0:
            return 0
        return np.max([rouge.get_scores(prediction, gt, avg=True)["rouge-l"]["f"] for gt in groundtruth])
    prediction = ' '.join(jieba.cut(prediction,use_paddle=True))
    groundtruth = ' '.join(jieba.cut(groundtruth,use_paddle=True))
    return rouge.get_scores(prediction, groundtruth, avg=True)["rouge-l"]["f"]


client = OpenAI(
            base_url="https://api.gpts.vin/v1",
            api_key="******"
        )
question='英国将中广核排除在核电站项目外，外交部称「望英方为中企提供公平非歧视的营商环境」，有哪些信息值得关注？'
prompt = f'请回答以下问题，注意需要以[x]标记出你引用参考资料的地方，比如[1]表示你引用了第一个参考资料，下面请回答:\n {question}'
messages=[
    {"role": "system", "content": "请参考以下资料回答问题"},
    {"role": "user", "content": prompt}
]
response=client.chat.completions.create(model="gpt-4-turbo", messages=messages)
gpt_answer = response.choices[0].message.content
webchat_result='英国政府的决定：英国政府决定将中国广核集团有限公司（中广核）排除在塞兹维尔C核电项目之外，而让英政府和法国电力集团共同持有该项目的股份[5]。 2. 英政府的动机和解释：尽管具体支付给中广核的金额未详细透露，英国政府明显付出了相当的金钱以确保中广核能退出该项目，包括收购成本和其他应付税项[5]。此举在英国媒体和相关报道中被视为英国与中国关系的进一步降温[1]。 3. 中国的反应：中国外交部发言人赵立坚表达了希望英方能为在英中国企业提供一个公平、公正、非歧视性的营商环境的期望[1]。此外，中国驻英国大使馆也确认相关企业已就此事签署协议[5]。 4. 反映更大的地缘政治背景：此事件发生在背景中，英方领导人曾指出中国对英国的利益和价值观构成了系统性的威胁[4]。此种说法反映了英国与中国在更宽广地缘政治和安全层面上的紧张关系。 5. 媒体报道与评论的角度：英国《金融时报》对于英国的这一行动解读为英中关系的进一步冷却[1]，同时报道英国对外国直接投资的策略可能正在调整，反映出对中国企业参与关键基础设施项目的警觉态度。[5]'

rouges = []
original = []
length = []
result={"question": "英国将中广核排除在核电站项目外，外交部称「望英方为中企提供公平非歧视的营商环境」，有哪些信息值得关注？", "abstracts": ["英国政府正在考虑如何将中广核剔除出英国未来所有的核电项目，包括其计划在萨福克郡斥资200亿英镑建造的塞兹韦尔核电站。报道还声称，此举有助于吸引北美投资者加入英国的核电项目。中广核是欣克利角C核电站33%的投资者，在萨福克郡的塞兹韦尔C核电项目中开发中将拥有20%的股份，并共同推进布拉德韦尔B核电项目，其计划在该厂安装自己的华龙HPR1000反应堆技术。英国政府不希望中广核参与这两个项目，但希望该公司能在没有任何对抗的情况下退出。将中广核从塞兹韦尔C项目剔除，或有助于法国电力集团为该项目吸引北美的基础设施投资者。", "7月6日，国务院总理李克强在中南海紫光阁同英国工商界代表举行视频对话会时就英国将中广核排除在核电站项目外一事明确表示，希望英方能为中国企业赴英投资兴业提供公平、公正、非歧视的营商环境。", "美国被曝要在中东修铁路，耶伦警告拜登：这样做可能引发宪法危机，“全球新闻界应该为阿桑奇辩护”，“这是对英国民主的无耻侮辱”，“15年来将首次预算盈余，与中国有关”，“巴基斯坦或将使用人民币购买俄油”，美国得州车辆冲入公交站候车人群，已致8死12伤，“因中国加入，年底前俄乌或将启动谈判”，无视美国警告，阿盟同意叙利亚回归，纽约地铁出现这一幕，“乌克兰领导人担心反攻不会达到西方期望”，美国佛州拟禁止中国公民买房购地，引发歧视争议，查尔斯三世加冕前夕，CNN民调“拆台”，“她们被韩政府献给美国大兵赚外汇，如今却被遗忘”。", "“关于你提到的塞兹韦尔C核电站，中国驻英使馆已就此作出了回应，我们尊重中方企业的决定。近年来，中英法在英开展了多个核电基础设施合作项目，总体持续推进。我们希望英方为中资企业提供公平、公正、非歧视的营商环境，为中英合作创造有利条件。”", "英国政府宣布，中国广核集团有限公司（中广核）将被排除参与塞兹韦尔C核电项目。中国外交部发言人赵立坚30日表示，希望英方为在英中国企业提供公平、公正、非歧视性的营商环境。报道称，英方此举意味着中广核被排除参与该项目，总预算为200亿英镑，中广核持有其中20%的股份。赵立坚表示，希望英方为中资企业提供公平、公正、非歧视的营商环境，为中英合作创造有利条件。", "针对苏纳克涉华表态，赵立坚11月30日在例行记者会上回应称，中国一贯奉行独立自主的和平外交政策，是各国发展的伙伴和机遇，不是威胁和挑战。中方始终致力于在相互尊重、互利共赢的基础上发展同各国的关系，同时坚决维护自身的主权、安全和发展利益。英方应摒弃冷战思维，停止散布“中国威胁论”，客观理性看待中国和中英关系。据报道，美国2019年将中广核列入出口管制黑名单时，美国政府曾警告英国不要让中国参与核电项目。"], "answer": "英国将中广核排除在核电站项目外，外交部称「望英方为中企提供公平非歧视的营商环境」，值得关注的信息有以下几个方面：\n\n首先，英国政府此举意味着中广核将被排除在200亿英镑的塞兹韦尔C核电项目及其他未来核电项目中，中国外交部发言人赵立坚表示，希望英方为中国企业赴英投资兴业提供公平、公正、非歧视的营商环境。此举有助于吸引北美投资者加入英国的核电项目，也有助于法国电力集团为该项目吸引北美的基础设施投资者。\n\n其次，此举可能与美国有关。据报道，美国2019年将中广核列入出口管制黑名单时，美国政府曾警告英国不要让中国参与核电项目。因此，英国此举可能是在响应美国的警告，也可能是为了避免在美国的制裁下受到影响。\n\n最后，英国此举可能会影响中英关系。中国外交部发言人赵立坚表示，希望英方摒弃冷战思维，停止散布“中国威胁论”，客观理性看待中国和中英关系。如果英国不能提供公平和非歧视的营商环境，可能会影响中英双边关系的发展。"}
r = get_rouge_over_list(gpt_answer, result['answer'])
rouges.append(r)

print("rouge score: {}".format(np.mean(rouges)))