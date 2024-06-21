import json
import os
import numpy as np


jsons = 'EED01.json', 'EED02.json', 'EED03.json'
merge_path = 'EED6000.json'
EED4LLM_path = 'EED4LLM_6000.json'
dialogue_train = np.load("data/ED/sys_dialog_texts.train.npy", allow_pickle=True)
situation_train = np.load("data/ED/sys_situation_texts.train.npy", allow_pickle=True)
emotion_train = np.load("data/ED/sys_emotion_texts.train.npy", allow_pickle=True)
target_train = np.load("data/ED/sys_target_texts.train.npy", allow_pickle=True)

input_example = {
            "User Query": "It's all because of the traffic jam. It was terrible and very frustrating.",
            # "Conversation Situation": "I really hate being late for work because of traffic",
            "Conversation History": "[User]I was late for work today.[Agent]Can you tell me what happened?",
        }
output_example = {
            "Emotion Cause": "Traffic jam.",
            "Event Scenario": "Work-related stress.",
            "Rationale": "Traffic jam results in lateness, causing individuals to feel anxious and frustrated.",
            "Goal to Response": "Alleviating anxiety and agitation.",
            "Agent Gender": "Female",
            "Agent Age": "Young adults",
            "Agent Timbre and Tone": "Soft",
            "Empathetic Reponse": "I can imagine how frustrating and challenging it must have been to deal with such a terrible traffic jam. It's understandable that it caused you to be late for work.",
            "Emotional Response": "Terrified"
        }

prompt = "You are an experienced and empathetic chatbot. Unlike regular chat systems, you are a virtual digital entity that adapts its gender, age, and voice to better serve different users. Now you are provided with a conversation which contains user's current queries and conversation history." \
         "Your task is firstly to comprehend the provided conversation information to the best of your ability, and then to proceed with a step-by-step, in-depth analysis following the procedure outlined below, and finally output your results of each step." \
         "Step 1: <Emotion Cause> The reason behind the user's emotional state." \
         "Step 2: <Event Scenario> The scenario in which the conversation takes place. You can focus on the events mentioned in the conversation or infer the scenario based on common sense and domain knowledge, such as daily conversation, psychological assistance, elder people company, or children company, etc. The result of this step is summary-oriented, ideally consisting of no more than 5 words and allowing for repetition." \
         "Step 3: <Rationale> The underlying reasons behind the user's emotions or the occurrence of the current event." \
         "Step 4: <Goal to Response> Determine the goal the chatbot should aim for in its response based on the analysis from the previous steps." \
         "Step 5: <Agent gender> The gender of the chatbot, selecting from male and female." \
         "Step 6: <Agent age> The age of the chatbot, selecting from Children, Teenagers, Young Adults, Middle-aged Adults and Elderly Adults." \
         "Step 7: <Agent Timbre and Tone> The timbre of the chatbot, selecting from Low-pitched, Soft, Clear, Melodious, warm, husky, bright." \
         "Step 8: <Empathetic Response> Leveraging the comprehensive analysis of the aforementioned steps, provide the user with an empathetic response to their query."\
         "Step 9: <Emotional Response> The emotional tone conveyed when replying to the user." \
         f"Here is an example, given the input: {input_example}. Your output must strictly comply with the JSON format and refrain from outputting any other irrelevant content, as shown in the following output example: {output_example}." \

def load_EED():
    with open(merge_path, 'r') as f:
        EED_dataset = json.load(f)
        str_EED = [json.dumps(data) for data in EED_dataset]  # trans dict to str
        print(f"Data Volume: {len(str_EED)}")
    return str_EED

def normalize_EED(eed):
    if eed.startswith("{") and eed.endswith("}"):
        normalized_eed = eed[1:-1]
    else:
        start_index = eed.find("{")
        end_index = eed.rfind("}")
        normalized_eed = eed[start_index+1:end_index-1]
    return normalized_eed
def merge_jsons(merge_path, *args):
    args = list(args)[0]
    EED_dataset = []
    for i, arg in enumerate(args):
        with open(arg, 'r') as f:
            EED_dataset.extend(json.load(f))
    # for j, item in enumerate(EED_dataset):
    #     if isinstance(item, str):
    #         try:
    #             EED_dataset[i] = json.loads(item)
    #         except json.decoder.JSONDecodeError:
    #             item = item.replace('"s', "'s")
    #
    #             if not item.startswith("{"):
    #                 start_index = item.find("{")
    #                 if start_index != -1:
    #                     item = item[start_index:]
    #                 else:
    #                     item = "{" + item
    #
    #             if not item.endswith("}"):
    #                 end_index = item.rfind("}")
    #                 if end_index != -1:
    #                     item = item[:end_index + 1]
    #                 else:
    #                     item = item + "}"
    #
    #             EED_dataset[i] = json.loads(item)
    #
    #         except Exception as e:
    #             print(f"Exception occurs at number {j}: {e}, Skipping！")
    with open(merge_path, 'w') as file:
        json.dump(EED_dataset, file, indent=4)

def buildforLLM(str_EED):
    EED4LLM_list = []
    for i, eed in enumerate(str_EED):
        response_target = target_train[i]
        emotion = emotion_train[i]
        if len(dialogue_train[i]) > 1:
            history = ""
            for dia_num in range(len(dialogue_train[i])-1):
                if dia_num % 2 == 0:
                    utt = '[user]' + dialogue_train[i][dia_num]
                else:
                    utt = '[agent]' + dialogue_train[i][dia_num]
                history += utt
        else:
            history = "None"
        user_query = dialogue_train[i][-1]
        context = f"{prompt} \n Now here is a input for you:\n User Query:{user_query} \n Conversation History:{history}\n "
        target = normalize_EED(eed) + f',"Empathetic Response":{response_target} "Emotional Response":{emotion}"'
        EED4LLM_list.append({'context': context, 'target': target})
    return EED4LLM_list

if __name__ == '__main__':
    # if os.path.exists(EED4LLM):
    #     with open(MERG_path, 'r') as f:
    #         EED4LLM = json.load(f)

    if os.path.exists(merge_path):
        str_EED = load_EED()
    else:
        merge_jsons(merge_path, jsons)
        str_EED = load_EED()
    EED4LLM = buildforLLM(str_EED)

    json_data = json.dumps(EED4LLM, indent=4)
    with open(EED4LLM_path, 'w') as file:
        file.write(json_data)