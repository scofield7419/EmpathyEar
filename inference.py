import time
import json
import re
import random
import yaml
import argparse
import torch
import os
from munch import Munch
import numpy as np
from torch import nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer
from peft import get_peft_model, LoraConfig, TaskType
from StyleTTS2.styletts2 import StyleTTS2
# import IPython.display as ipd
from scipy.io import wavfile
from EAT.demo import EAT
from EAT.preprocess.deepspeech_features.extract_ds_features import extract_features

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
emotion_aligned = {
                    "suprised":"sur",
                    "excited":"hap",
                    "angry":"ang",
                    "proud":"hap",
                    "sad":"sad",
                    "annoyed":"ang",
                    "grateful":"hap",
                    "lonely":"sad",
                    "afraid":"fea",
                    "terrified":"fea",
                    "guilty":"sad",
                    "impressed":"sur",
                    "disgusted":"dis",
                    "hopeful":"hap",
                    "confident":"neu",
                    "furious":"ang",
                    "anxious":"sad",
                    "anticipating":"hap",
                    "joyful":"hap",
                    "nostalgic":"sad",
                    "disappointed":"sad",
                    "prepared":"neu",
                    "jealous":"ang",
                    "content":"hap",
                    "devastated":"sur",
                    "embarrassed":"neu",
                    "caring":"hap",
                    "sentimental":"sad",
                    "trusting":"neu",
                    "ashamed":"neu",
                    "apprehensive":"fea",
                    "faithful":"neu",
}
                
prompt = "You are an experienced and empathetic chatbot. Unlike regular chat systems, you are a virtual digital entity that adapts its gender, age, and voice to better serve different users. Now you are provided with a conversation which contains user's current queries and conversation history.Your task is firstly to comprehend the provided conversation information to the best of your ability, and then to proceed with a step-by-step, in-depth analysis following the procedure outlined below, and finally output your results of each step.Step 1: <Emotion Cause> The reason behind the user's emotional state.Step 2: <Event Scenario> The scenario in which the conversation takes place. You can focus on the events mentioned in the conversation or infer the scenario based on common sense and domain knowledge, such as daily conversation, psychological assistance, elder people company, or children company, etc. The result of this step is summary-oriented, ideally consisting of no more than 5 words and allowing for repetition.Step 3: <Rationale> The underlying reasons behind the user's emotions or the occurrence of the current event.Step 4: <Goal to Response> Determine the goal the chatbot should aim for in its response based on the analysis from the previous steps.Step 5: <Agent gender> The gender of the chatbot, selecting from male and female.Step 6: <Agent age> The age of the chatbot, selecting from Children, Teenagers, Young Adults, Middle-aged Adults and Elderly Adults.Step 7: <Agent Timbre and Tone> The timbre of the chatbot, selecting from Low-pitched, Soft, Clear, Melodious, warm, husky, bright.Step 8: <Empathetic Response> Leveraging the comprehensive analysis of the aforementioned steps, provide the user with an empathetic response to their query.Step 9: <Emotional Response> The emotional tone conveyed when replying to the user.Here is an example, given the input: {'User Query': \"It's all because of the traffic jam. It was terrible and very frustrating.\", 'Conversation History': '[User]I was late for work today.[Agent]Can you tell me what happened?'}. Your output must strictly comply with the JSON format and refrain from outputting any other irrelevant content, as shown in the following output example: {'Emotion Cause': 'Traffic jam.', 'Event Scenario': 'Work-related stress.', 'Rationale': 'Traffic jam results in lateness, causing individuals to feel anxious and frustrated.', 'Goal to Response': 'Alleviating anxiety and agitation.', 'Agent Gender': 'Female', 'Agent Age': 'Young adults', 'Agent Timbre and Tone': 'Soft', 'Empathetic Reponse': \"I can imagine how frustrating and challenging it must have been to deal with such a terrible traffic jam. It's understandable that it caused you to be late for work.\", 'Emotional Response': 'Terrified'}. \n"

def get_all_filenames(path):
    filenames = []
    for root, dirs, files in os.walk(path):
        for file in files:
            filenames.append(path + file)
    return filenames

def read_npy(npy_path):
    dialogue_list = []
    data = np.load(npy_path, allow_pickle=True)
    for i, dia in enumerate(data):
        history = ""
        if len(data) > 1:
            for dia_num in range(len(data[i])-1):
                if dia_num % 2 == 0:
                    utt = '[user]' + dia[dia_num]
                else:
                    utt = '[agent]' + dia[dia_num]
                history += utt
        else:
            history = "None"
        user_query = data[i][-1]
        dialogue_list.append(f"User Query:{user_query} \n Conversation History:{history}\n ")
    return dialogue_list

def add_quotes_if_missing(input_string):
    search_list = ['"Empathetic Response":', '"Emotional Response":']
    quote = "\""
    output_string = input_string
    # 查找目标字符串
    for search_string in search_list:
        index = output_string.find(search_string)
        if index != -1:
            # 判断目标字符串后面是否有双引号
            if output_string[index + len(search_string)] != quote:
                # 在目标字符串后面添加双引号
                output_string = output_string[:index + len(search_string)] + quote + output_string[index + len(search_string):]
            if output_string[index-1] != quote:
                output_string = output_string[:index-1] + quote + ',' + output_string[index:]
    return output_string

def main():
    torch.manual_seed(0)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    random.seed(0)
    np.random.seed(0)

    # Argument Parser Setup
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str,
                    default="chatglm/chatglm-6b-base",
                    help="The directory of the model")
    parser.add_argument("--tokenizer", type=str, default="chatglm/chatglm-6b-base", help="Tokenizer path")
    parser.add_argument("--LoRA", type=str, default=True, help="use lora or not")
    parser.add_argument("--lora-path", type=str,default='chatglm/checkpoint-24000/pytorch_model.pt',help="Path to the LoRA model checkpoint")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device to use for computation")
    parser.add_argument("--max-new-tokens", type=int, default=2048, help="Maximum new tokens for generation")
    parser.add_argument("--lora-alpha", type=float, default=32, help="LoRA alpha")
    parser.add_argument("--lora-rank", type=int, default=8, help="LoRA r")
    parser.add_argument("--lora-dropout", type=float, default=0.1, help="LoRA dropout")
    parser.add_argument("--test-data", type=str, default="data/empathetic-dialogue-emb/sys_dialog_texts.test.npy")
    parser.add_argument("--wav_save_path", type=str, default="TTS_results/ED_test/")
    parser.add_argument("--mp4_save_path", type=str, default="MP4_results/ED_test/")
    parser.add_argument("--driven_video", type=str, default="EAT/demo/video_processed/template")
    args = parser.parse_args()

    if args.tokenizer is None:
        args.tokenizer = args.model

    if args.LoRA:
    # Model and Tokenizer Configuration
        chatglm_tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)
        chatglm_model = AutoModel.from_pretrained(args.model, load_in_8bit=False, trust_remote_code=True, device_map="auto")
    
        # LoRA Model Configuration
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM, inference_mode=True,
            target_modules=['query_key_value'],
            r=args.lora_rank, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout
        )
        chatglm_model = get_peft_model(chatglm_model, peft_config)
        if os.path.exists(args.lora_path):
            chatglm_model.load_state_dict(torch.load(args.lora_path), strict=False)
    
    else:
        chatglm_tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)
        chatglm_model = AutoModel.from_pretrained(args.model, load_in_8bit=False, trust_remote_code=True, device_map="auto")
    
    test_data = read_npy(args.test_data)
    

    for num, test_dia in enumerate(test_data):
        wav_save_path = args.wav_save_path + f'test{num+1}'
        mp4_save_path = args.mp4_save_path + f'test{num+1}'
        for folder_path in [wav_save_path, mp4_save_path]:
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)
                print(f"Folder '{folder_path}' created.")
                continue_outer = False
            else:
                print(f"Folder '{folder_path}' already exists.")
                continue_outer = True
        if continue_outer:
            continue
        os.makedirs(wav_save_path + '/deepfeature32')
        # empathetic response text generation
        # prompt = "You are an experienced and empathetic chatbot. Unlike regular chat systems, you are a virtual digital entity that adapts its gender, age, and voice to better serve different users. Now you are provided with a conversation which contains user's current queries and conversation history.Your task is firstly to comprehend the provided conversation information to the best of your ability, and then to proceed with a step-by-step, in-depth analysis following the procedure outlined below, and finally output your results of each step."\
        #          "Step 1: <Emotion Cause> The reason behind the user's emotional state."\
        #          "Step 2: <Event Scenario> The scenario in which the conversation takes place. You can focus on the events mentioned in the conversation or infer the scenario based on common sense and domain knowledge, such as daily conversation, psychological assistance, elder people company, or children company, etc. The result of this step is summary-oriented, ideally consisting of no more than 5 words and allowing for repetition."\
        #          "Step 3: <Rationale> The underlying reasons behind the user's emotions or the occurrence of the current event."\
        #          "Step 4: <Goal to Response> Determine the goal the chatbot should aim for in its response based on the analysis from the previous steps."\
        #          "Step 5: <Agent gender> The gender of the chatbot, selecting from male and female."\
        #          "Step 6: <Agent age> The age of the chatbot, selecting from Children, Teenagers, Young Adults, Middle-aged Adults and Elderly Adults."\
        #          "Step 7: <Agent Timbre and Tone> The timbre of the chatbot, selecting from Low-pitched, Soft, Clear, Melodious, warm, husky, bright."\
        #          "Step 8: <Empathetic Response> Leveraging the comprehensive analysis of the aforementioned steps, provide the user with an empathetic response to their query."\
        #          "Step 9: <Emotional Response> The emotional tone conveyed when replying to the user."\
        
        cur_prompt = prompt + f"Now here is an input for you:{test_dia}"

        #response:I can see why you're frustrated with the lack of security in your neighborhood. It's disheartening when the police aren't able to protect the community.
        chatglm_inputs = chatglm_tokenizer(cur_prompt, return_tensors="pt").to(args.device)
        response = chatglm_model.generate(input_ids=chatglm_inputs["input_ids"],
                            max_length=chatglm_inputs["input_ids"].shape[-1] + args.max_new_tokens)
        response = response[0, chatglm_inputs["input_ids"].shape[-1]:]
        text_response = chatglm_tokenizer.decode(response, skip_special_tokens=True)
        print(text_response)
        print("************************************************************************************************")

        try:
            # # 移除首尾的引号
            text_response = add_quotes_if_missing(text_response)

            # 将字符串解析为字典
            pattern =  r'"([^"]+)"\s*:\s*([^,]+)'
            matches = re.findall(pattern, text_response)
            data = {key: value for key, value in matches}
            emotion_cause = data["Emotion Cause"].strip('"')
            event_scenario =  data["Event Scenario"].strip('"')
            rationale = data["Rationale"].strip('"')
            goal_to_response = data["Goal to Response"].strip('"')
            agent_gender = data["Agent Gender"].strip('"')
            agent_age = data["Agent Age"].strip('"')
            agent_timbre_tone = data["Agent Timbre and Tone"].strip('"')
            empathetic_response = data["Empathetic Response"].strip('"')
            if "Emotional Response" in data:
                emotional_response = data["Emotional Response"].strip('"')
                try:
                    emotion_type = emotion_aligned[emotional_response]
                except:
                    emotion_type = "neu"
            else:
                emotional_response = "neu"
        except:
            continue


        #TTS for response text
        tts = StyleTTS2()
        if agent_gender=='Female':
            wav_file = "StyleTTS2/Demo/reference_audio/W/" + agent_timbre_tone.lower() + '.wav'
        elif agent_gender=='Male':
            wav_file = "StyleTTS2/Demo/reference_audio/M/" + agent_timbre_tone.lower() + '.wav'
        # Wav_path = "StyleTTS2/Demo/EED_wav/"
        # Wav_dict = get_all_filenames(Wav_path)
        # for wav_file in Wav_dict:
        result_name = wav_file.split('/')[-1]
        start = time.time()
        noise = torch.randn(1, 1, 256).to(args.device)
        ref_s = tts.compute_style(wav_file)
        wav = tts.inference(empathetic_response, ref_s, alpha=0.3, beta=0.7, diffusion_steps=5, embedding_scale=1)
        rtf = (time.time() - start) / (len(wav) / 24000)
        print(f"RTF = {rtf:5f}")
        # print(k + ' Synthesized:')
        # display(ipd.Audio(wav, rate=24000, normalize=False))
        # print('Reference:')
        # display(ipd.Audio(path, rate=24000, normalize=False))
        scaled_data = np.int16(wav * 32767)
        wav_file_path = wav_save_path + '/' + result_name
        wavfile.write(wav_file_path, 24000, scaled_data)


        #extract wav deepspeech features
        extract_features(wav_save_path, wav_save_path + '/deepfeature32')
        #wav2talkingface
        eat = EAT(root_wav=wav_save_path)
        eat.tf_generate(agent_age, agent_gender, emotion_type, save_dir=mp4_save_path)


if __name__ == "__main__":
    main()



