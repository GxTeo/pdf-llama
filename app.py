import gradio as gr
from gradio_pdf import PDF
import fitz
from PIL import Image
from model import Process
import time


def check_pdf(pdf):
    if(pdf):
        return True

history = []
def respond(query: str, chat_history, pdf):
    global history
    if not query:
         raise gr.Error('Please enter your query')
    if not check_pdf(pdf):
        raise gr.Error(message='Upload a PDF')

    process = Process(pdf)
    qa_chain = process.setup_chain()

    result = qa_chain({'question': query, 'chat_history': history})
    bot_answer = result['answer']

    history.append((query, bot_answer))
    chat_history.append((query, bot_answer))

    time.sleep(1)
    return chat_history

def clear_textbox():
    return gr.update(value="")


with gr.Blocks() as demo:
    with gr.Row():
        with gr.Column(scale=1):
            chatbot = gr.Chatbot(value=[], height='90vh')
        with gr.Column(scale=1):
            pdf = PDF(label="Upload a PDF", interactive=True, height=680, container=True,scale=1)
    with gr.Row():
        query = gr.Textbox(show_label=False, placeholder='Enter your query and click submit', container=False, lines=3, scale=0.8)
        # Submit query
        submit_btn = gr.Button('submit', scale=0.2)

    submit_btn.click(
            fn=respond, 
            inputs=[query, chatbot, pdf], 
            outputs=[chatbot, ], 
            queue=False).then(clear_textbox, None, [query], queue=False)
    
    demo.launch(debug=True)