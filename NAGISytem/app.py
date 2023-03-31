from transformers import AutoTokenizer, AutoModelForMaskedLM
import torch
BERTTokenizer = AutoTokenizer.from_pretrained("cl-tohoku/bert-base-japanese")
BERTModel = AutoModelForMaskedLM.from_pretrained("cl-tohoku/bert-base-japanese")

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
mT5Tokenizer = AutoTokenizer.from_pretrained("google/mt5-base")
mT5Model = AutoModelForSeq2SeqLM.from_pretrained("google/mt5-base")

from transformers import AutoTokenizer, AutoModelForCausalLM
GPT2Tokenizer = AutoTokenizer.from_pretrained("rinna/japanese-gpt2-medium")
GPT2Model = AutoModelForCausalLM.from_pretrained("rinna/japanese-gpt2-medium")

import gradio as gr

votes=[]
BERT=None
mT5=None
GPT2=None
def MELCHIOR(sue):
    #BERT
    allow=BERTTokenizer("承認").input_ids[1]
    deny=BERTTokenizer("否定").input_ids[1]
    output=BERTModel(**BERTTokenizer('科学者としての人格を持ったMELCHIORは次の決議に答えます。人間「'+sue+'承認か否定どちらですか？」'+"MELCHIOR 「[MASK]」",return_tensors="pt")).logits
    BERTTokenizer.batch_decode(torch.argmax(output,-1))
    mask=output[0,-3,:]
    votes.append(1 if mask[allow]>mask[deny] else -1)
    return "承認"  if mask[allow]>mask[deny] else "否定"

def BALTHASAR(sue):
    #mT5
    allow=mT5Tokenizer("承認").input_ids[1]
    deny=mT5Tokenizer("否定").input_ids[1]
    encoder_output=mT5Model.encoder(**mT5Tokenizer('母としての人格を持ったBALTHASARは次の決議に答えます。人間「'+sue+'承認か否定どちらですか？」'+"BALTHASAR 「<X>」",return_tensors="pt"))
    id=None
    p_answer=None
    probs=None
    i=0
    txt="<pad>"
    probs=mT5Model(inputs_embeds=encoder_output.last_hidden_state,decoder_input_ids=mT5Tokenizer(txt,return_tensors="pt").input_ids).logits[0]
    id=torch.argmax(probs[i+1])
    txt=txt+"<X>"
    i=i+1
    probs=mT5Model(inputs_embeds=encoder_output.last_hidden_state,decoder_input_ids=mT5Tokenizer(txt,return_tensors="pt").input_ids).logits[0]
    id=torch.argmax(probs[i+1])
    txt=txt+mT5Tokenizer.decode(id)
    votes.append(1 if probs[i+1][allow]>probs[i+1][deny] else -1)
    return "承認"  if probs[i+1][allow]>probs[i+1][deny] else "否定"

def CASPER(sue):
    #GPT2
    allow=GPT2Tokenizer("承認").input_ids[1]
    deny=GPT2Tokenizer("否定").input_ids[1]
    probs=GPT2Model(**GPT2Tokenizer('女としての人格を持ったCASPERは次の決議に答えます。人間「'+sue+'承認か否定どちらですか？」'+"CASPER 「",return_tensors="pt")).logits[0]
    i=0
    p_answer=probs
    id=torch.argmax(probs[0])
    votes.append(1 if probs[0][allow]>probs[1][deny] else -1)
    return "承認" if probs[0][allow]>probs[1][deny] else "否定"

def greet(sue):
    text1="BERT-1"+MELCHIOR(sue)
    text2="GPT-2"+CASPER(sue)
    text3="mt5-3"+BALTHASAR(sue)
    return text1+" "+text2+" "+text3+"\n                 ____\n\n"+("                |可決|" if sum(votes[-3:])>0 else "                |否決|")+"\n                 ____"

css=".gradio-container {background-color: black} .gr-button {background-color: blue;color:black; weight:200%;font-family:YuMincho}.block{color:orange;} .gr-box {font-size: 140%;border-color:orange;background-color: #000000;weight:200%;font-family:YuMincho}"
with gr.Blocks(css=css) as demo:
    sue = gr.Textbox(label="NAGI System",placeholder="ここに決議内容を入力し，提訴を押してください．")
    greet_btn = gr.Button("提訴")
    output = gr.Textbox(label="決議")
    greet_btn.click(fn=greet, inputs=sue, outputs=output)
demo.launch()
