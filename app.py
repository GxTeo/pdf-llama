import gradio as gr

with gr.Blocks() as demo:

    with gr.Column():
        with gr.Row():
            with gr.Column(scale=0.8):
                api_key = gr.Textbox(
                    placeholder='Enter OpenAI API key',
                    show_label=False,
                    interactive=True
                ).style(container=False)
            with gr.Column(scale=0.2):
                change_api_key = gr.Button('Change Key')

        with gr.Row():
            chatbot = gr.Chatbot(value=[], elem_id='chatbot').style(height=650)
            show_img = gr.Image(label='Upload PDF', tool='select').style(height=680)

        
        with gr.Row():
            with gr.Column(scale=0.70):
                txt = gr.Textbox(
                    show_label=False,
                    placeholder="Enter your query"
                ).style(container=False)

            with gr.Column(scale=0.15):
                submit_btn = gr.Button('Submit')

            with gr.Column(scale=0.15):
                btn = gr.UploadButton("📁 Upload a PDF", file_types=[".pdf"]).style()

if __name__ == '__main__':
    demo.launch(debug=True)