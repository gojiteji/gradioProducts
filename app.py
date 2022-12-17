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

def greet(sue):
    votes=[]
    #BERT
    allow=BERTTokenizer("承認").input_ids[1]
    deny=BERTTokenizer("否定").input_ids[1]
    output=BERTModel(**BERTTokenizer('科学者としての人格を持ったMELCHIORは次の決議に答えます。人間「'+sue+'承認か否定どちらですか？」'+"MELCHIOR 「[MASK]」",return_tensors="pt")).logits
    BERTTokenizer.batch_decode(torch.argmax(output,-1))
    mask=output[0,-3,:]
    print("actual token:",BERTTokenizer.decode(torch.argmax(mask)))
    print(BERTTokenizer.decode(allow if mask[allow]>mask[deny] else deny))
    votes.append(1 if mask[allow]>mask[deny] else -1)

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

    #GPT
    allow=GPT2Tokenizer("承認").input_ids[1]
    deny=GPT2Tokenizer("否定").input_ids[1]
    probs=GPT2Model(**GPT2Tokenizer('女としての人格を持ったCASPERは次の決議に答えます。人間「'+sue+'承認か否定どちらですか？」'+"CASPER 「",return_tensors="pt")).logits[0]
    i=0
    p_answer=probs
    id=torch.argmax(probs[0])
    votes.append(1 if probs[1][allow]>probs[1][deny] else -1)
    return "可決" if sum(votes)>0 else "否決"

with gr.Blocks() as demo:
    sue = gr.Textbox(label="Magi System",placeholder="決議内容を入力してください")
    output = gr.Textbox(label="決議")
    greet_btn = gr.Button("提訴")
    greet_btn.click(fn=greet, inputs=sue, outputs=output)

demo.launch()
