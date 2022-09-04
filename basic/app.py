from transformers import pipeline
import gradio as gr
title = "Συμπλήρωση κειμένου με το μοντέλο GPT2"
description = "Ένα απλό gradio demo για να δούμε πως φτιάχνουμε εύκολα A.I. εφαρμογές με την χρήση των Pipelines"
examples = [
    ["Mike was the famous space mouse"],
    ["The Earth's perimeter is"],
    ["You will never believe what happened yesterday on my way back home."],
]

model = pipeline("text-generation" , model="gpt2")


def predict(prompt):
    completion = model(prompt)[0]["generated_text"]
    return completion

gr.Interface(
    fn=predict, 
    inputs="text", 
    outputs="text",
    title=title,
    description=description,
    examples=examples,
    enable_queue=True,
    ).launch()
