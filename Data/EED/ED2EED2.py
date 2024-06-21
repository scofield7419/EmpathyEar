import os
from json import JSONDecodeError

import openai
import numpy as np
import json
import time
import random
import torch
from tqdm import tqdm
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()
random.seed(13)
retry_limit = 5
sleep_time = 10
### initial prompt

# prompt = "This is a task related to empathetic dialogue, a complete empathetic dialogue consists of the following components:" \
#          "1. The situation in which the conversation occurs and the reference emotion of the conversation. " \
#          "2. A multi-round conversation that takes place between a speaker and a listener, the emotion label and situation " \
#          "of Speaker are invisible to Listener. Listener recognize and acknowledge Speaker’s feelings as much " \
#          "as possible and provides empathetic responses."\
#          "Please thoroughly analyze the entire dialogue and assign an emotional label to each " \
#          "utterrance from {neutral, joy, surprise, sadness, disgust, anger, fear}. In addition, to the best of your " \
#          "knowledge, you need to first imagine a reasonable scene in which the given dialogue took place and the identities " \
#          "of both the speaker and listener in the dialogue, and then speculate on the gender from {male，female} and approximate age " \
#          "from {young, middle, old} of the speaker and listener."\
#          "Here is an example:" \
#          "user: {" \
#          "context: I am very happy to have been first over 300 students during this years at my enginering school" \
#          "reference emotion: joyful" \
#          "dialogue: " \
#          "{" \
#          "1.Speaker: Hi,this year, I was the first over 300 students at my enginering school" \
#          "2.Listener: Sounds great! So what's your major?" \
#          "3.Speaker: It is computer science. I am very happy of this achievement and my family is very proud." \
#          "4.Listener: Well pleased. You should be having brains,man!That's a tough course, i hear." \
#          "}" \
#          "}" \
#          "gpt: {" \
#          "1:joy, 2:joy, 3:joy, 4:joy" \
#          "Speaker gender: young female" \
#          "Listener gender: young male." \
#          "}"
client = openai.OpenAI(
    api_key=os.environ.get("OPENAI_API_KEY"),
)

input_example = {
            "User Query": "It's all because of the traffic jam. It was terrible and very frustrating.",
            "Conversation History": "[User]I was late for work today.[Agent]Can you tell me what happened?",
            "Emotion": "angry",
        }
output_example = {
            "Emotion Cause": "Traffic jam.",
            "Event Scenario": "Work-related stress.",
            "Rationale": "Traffic jam results in lateness, causing individuals to feel anxious and frustrated.",
            "Goal to Response": "Alleviating anxiety and agitation.",
            "Agent Gender": "Female",
            "Agent Age": "Young adults",
            "Agent Timbre and Tone": "Soft",
        }

prompt = "You are an experienced chatbot. Unlike regular chat systems, you are a virtual digital entity that adapts its gender, age, and voice to better serve different users. Now you are provided with a conversation which contains user queries, conversation history, and the current user's emotional labels." \
         "Your task is firstly to comprehend the provided conversation information to the best of your ability, and then to proceed with a step-by-step, in-depth analysis following the procedure outlined below, and output the result for each step." \
         "Step 1: <Emotion Cause> The reason behind the user's emotional state." \
         "Step 2: <Event Scenario> The scenario in which the conversation takes place. You can focus on the events mentioned in the conversation or infer the scenario based on common sense and domain knowledge, such as daily conversation, psychological assistance, elder people company, or children company, etc. The result of this step is summary-oriented, ideally consisting of no more than 5 words and allowing for repetition." \
         "Step 3: <Rationale> The underlying reasons behind the user's emotions or the occurrence of the current event." \
         "Step 4: <Goal to Response> Determine the goal the chatbot should aim for in its response based on the analysis from the previous steps." \
         "Step 5: <Agent gender> The gender of the chatbot, selecting from male and female." \
         "Step 6: <Agent age> The age of the chatbot, selecting from Children, Teenagers, Young Adults, Middle-aged Adults and Elderly Adults." \
         "Step 7: <Agent Timbre and Tone> The timbre of the chatbot, selecting from Low-pitched, Soft, Clear, Melodious, warm, husky, bright." \
         f"Here is an example, given the input: {input_example}. Your output must strictly comply with the JSON format and refrain from outputting any other irrelevant content, as shown in the following output example: {output_example}." \

def get_res_chatgpt(contents):
    response = client.chat.completions.create(
        model="gpt-4-turbo-preview",
        # response_format='json_object',
        messages=contents,
    ).choices[0].message.content
    return response

def get_conv(input_set):
    context_list=[]
    context_list.append({
        'role': 'system',
        'content': prompt
    })
    context_list.append({
        'role': 'user',
        'content': input_set
    })
    for attempt in range(retry_limit):
        try:
            response = get_res_chatgpt(context_list)
            return response
        except (openai.APITimeoutError, openai.RateLimitError, openai.APIConnectionError, openai.BadRequestError):
            if attempt < retry_limit - 1:
                time.sleep(sleep_time)
                continue
            else:
                raise
        except Exception as e:
            print(f"发生异常：{type(e).__name__} - {str(e)}")

def build_input(dialogue, situation, target, emotion):
    if len(dialogue) > 1:
        history = ""
        for dia_num in range(len(dialogue)-1):
            if dia_num % 2 == 0:
                utt = '[user]' + dialogue[dia_num]
            else:
                utt = '[agent]' + dialogue[dia_num]
            history += utt
    else:
        history = "None"
    user_query = '[user]' + dialogue[len(dialogue)-1] + '[agent]' + target
    return f"User Query:{user_query}, Conversation History:{history}, Emotion:{emotion}"

def data_cleaning(json_path):
    with open(json_path, 'r') as f:
        EED_dataset = json.load(f)

    clean_data = []
    for num, d in enumerate(EED_dataset):
        try:
            clean_data.append(json.loads(d))
        except json.JSONDecodeError:
            clean_data.append(d.replace("'", "\""))
        except Exception as e:
            print(f"Exception occurs at number {num}: {e}, Skipping！")

    json_data = json.dumps(clean_data, indent=4)
    with open(json_path, 'w') as file:
        file.write(json_data)


def main():
    dialogue_train = np.load("data/ED/sys_dialog_texts.train.npy", allow_pickle=True)
    situation_train = np.load("data/ED/sys_situation_texts.train.npy", allow_pickle=True)
    emotion_train = np.load("data/ED/sys_emotion_texts.train.npy", allow_pickle=True)
    target_train = np.load("data/ED/sys_target_texts.train.npy", allow_pickle=True)
    len_train = len(dialogue_train)
    json_path = "EED02.json"

    if os.path.exists(json_path):
        with open(json_path, 'r') as f:
            EED_dataset = json.load(f)
    else:
        EED_dataset = []

    volume = 4000
    for i in tqdm(range(1999+len(EED_dataset), volume), desc="Task Progress"):
        dialogue = dialogue_train[i]
        situation = situation_train[i]
        emotion = emotion_train[i]
        target = target_train[i]

        input_set = build_input(dialogue, situation, target, emotion)
        gpt_response = get_conv(input_set)
        print(f"{i+1}:" + gpt_response)
        EED_dataset.append(gpt_response)
        if (i+1) % 100 == 0 or i == volume-1:
            json_data = json.dumps(EED_dataset, indent=4)
            with open(json_path, 'w') as file:
                file.write(json_data)
            print("**********************************\n")
            print(f'{i+1} has saved!')
            print("**********************************\n")
    print(f"{volume} data has generated, start data cleaning!")
    data_cleaning(json_path)

if __name__ == '__main__':
    main()

