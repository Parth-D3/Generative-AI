import torch
import gradio as gr
# Use a pipeline as a high-level helper
from transformers import pipeline

model_path = r"models\models--sshleifer--distilbart-cnn-12-6\snapshots\a4f8f3ea906ed274767e9906dbaede7531d660ff"
text_summary = pipeline("summarization", model=model_path, 
                torch_dtype=torch.bfloat16)

# text = """The Ford Mustang is a series of American automobiles manufactured by Ford. In continuous production since 1964, the Mustang is currently the longest-produced Ford car nameplate. Currently in its seventh generation, it is the fifth-best selling Ford car nameplate. The namesake of the "pony car" automobile segment, the Mustang was developed as a highly styled line of sporty coupes and convertibles derived from existing model lines, initially distinguished by "long hood, short deck" proportions.[3]

# Originally predicted to sell 100,000 vehicles yearly, the 1965 Mustang became the most successful vehicle launch since the 1927 Model A.[4] Introduced on April 17, 1964[5] (16 days after the Plymouth Barracuda), over 400,000 units were sold in its first year; the one-millionth Mustang was sold within two years of its launch.[6] In August 2018, Ford produced the 10-millionth Mustang; matching the first 1965 Mustang, the vehicle was a 2019 Wimbledon White convertible with a V8 engine.[7]
# The success of the Mustang launch led to multiple competitors from other American manufacturers, including the Chevrolet Camaro and Pontiac Firebird[8] (1967), AMC Javelin (1968), and Dodge Challenger[9] (1970). It also competed with the Plymouth Barracuda, which was launched around the same time. The Mustang also had an effect on designs of coupes worldwide, leading to the marketing of the Toyota Celica and Ford Capri in the United States (the latter, by Lincoln-Mercury). The Mercury Cougar was launched in 1967 as a unique-bodied higher-trim alternative to the Mustang; during the 1970s, it included more features and was marketed as a personal luxury car."""
# print(text_summary(text))

def Summary(input):
    output = text_summary(input)

    return output[0]['summary_text']

gr.close_all()

# demo = gr.Interface(fn=Summary,inputs="text",outputs="text")
demo = gr.Interface(fn=Summary,
                    inputs=[gr.Textbox(label="Input text to summarize",lines=6)],
                    outputs=[gr.Textbox(label="Summarized text",lines=4)],
                    title="Text Summarizer",
                    description="THIS APPLICATION WILL BE USED TO SUMMARIZE THE TEXT")
demo.launch()




