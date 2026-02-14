import gradio as gr
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-medium")

dados_pedidos = {
    "numero_pedido": ["20234", "39873", "10945", "93078"],
    "status": ["Shipped", "Processing", "Delivered", "Cancelled"]
}

df_status_pedidos = pd.DataFrame(dados_pedidos)

def verificar_status_pedido(numero_pedido): 
  try:
    status = df_status_pedidos[df_status_pedidos["numero_pedido"] == numero_pedido]['status'].iloc[0]

    return f'The status of your order {numero_pedido} is {status}'

  except:
    return f'There is no order with the number {numero_pedido}. Please check and try again.'

palavras_chave_status = ['order', 'order status', 'status of my order', 'check my order', 'order update']

def responder(input_usuario, ids_historico_chat):
  if any(keyword in input_usuario.lower() for keyword in palavras_chave_status):
    return 'Please enter your order number: ', ids_historico_chat
  else:
    novo_usuario_input_ids = tokenizer.encode(input_usuario + tokenizer.eos_token, return_tensors='pt')

    if ids_historico_chat is not None:
      bot_input_ids = torch.cat([ids_historico_chat, novo_usuario_input_ids], dim=-1)

    else:
      bot_input_ids = novo_usuario_input_ids

    ids_historico_chat = model.generate(
        bot_input_ids, 
        max_length=1000, 
        pad_token_id=tokenizer.eos_token_id
        )
    resposta_bot = tokenizer.decode(ids_historico_chat[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)

  return resposta_bot, ids_historico_chat

with gr.Blocks() as app:
  chatbot = gr.Chatbot()
  msg = gr.Textbox(placeholder='Type your message here...')
  
  estado = gr.State(None)
  aguardando_numero_pedido = gr.State(False)

  def processar_entrada(input_usuario, historico, ids_historico_chat, aguardando_numero_pedido):
    if aguardando_numero_pedido:
      resposta = verificar_status_pedido(input_usuario)
      aguardando_numero_pedido = False
    else:
      resposta, ids_historico_chat = responder(input_usuario, ids_historico_chat)
      if resposta == 'Please enter your order number: ':
        aguardando_numero_pedido = True
    
    historico.append((input_usuario, resposta))
    return historico, ids_historico_chat, aguardando_numero_pedido, ""

  msg.submit(
      processar_entrada,
      [msg, chatbot, estado, aguardando_numero_pedido],
      [chatbot, estado, aguardando_numero_pedido, msg]
  )
  
  if "__name"=="__main":  
      app.launch(share=True)