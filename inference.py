import time
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
import IPython.display as ipd
from scipy.io import wavfile


def main():
    torch.manual_seed(0)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    random.seed(0)
    np.random.seed(0)

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str,
                    default="xxx",
                    help="The directory of the model")
    parser.add_argument("--tokenizer", type=str, default="xxx", help="Tokenizer path")
    parser.add_argument("--LoRA", type=str, default=False, help="use lora or not")
    parser.add_argument("--lora-path", type=str,default=False,help="Path to the LoRA model checkpoint")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use for computation")
    parser.add_argument("--max-new-tokens", type=int, default=512, help="Maximum new tokens for generation")
    parser.add_argument("--lora-alpha", type=float, default=32, help="LoRA alpha")
    parser.add_argument("--lora-rank", type=int, default=8, help="LoRA r")
    parser.add_argument("--lora-dropout", type=float, default=0.1, help="LoRA dropout")

    args = parser.parse_args()

    if args.tokenizer is None:
        args.tokenizer = args.model

    if args.LoRA:
    # Model and Tokenizer Configuration
        chatglm_tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)
        chatglm_model = AutoModel.from_pretrained(args.model, load_in_8bit=False, trust_remote_code=True, device_map="auto").to(args.device)

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
        chatglm_model = AutoModel.from_pretrained(args.model, load_in_8bit=False, trust_remote_code=True, device_map="auto").to(args.device)


    # empathetic response text generation
    prompt = "You are an empathetic listener, your task is give an empathetic response based on the context of the conversation." \
             "Additionally, attach an emotion label from {neutral, joy, surprise, sadness, disgust, anger, fear} for your empathetic response." \
             "For example, " \
             "{" \
             "conversation:" \
             "<speaker: Today traffic was horrible and was so frustrating!>" \
             "empathetic response: I hate traffic too, it makes me angry!" \
             "emotion label: angry" \
             "}" \
             "Now give your empathetic response and it's emotion label for this conversation:" \
             "<speaker: I have been awarded with a degree in marketing.>"
    chatglm_inputs = chatglm_tokenizer(prompt, return_tensors="pt").to(args.device)
    response = chatglm_model.generate(input_ids=chatglm_inputs["input_ids"],
                          max_length=chatglm_inputs["input_ids"].shape[-1] + args.max_new_tokens)
    response = response[0, chatglm_inputs["input_ids"].shape[-1]:]
    text_response = chatglm_tokenizer.decode(response, skip_special_tokens=True)
    print("Response:", text_response)

    #TTS for response text
    text = text_response
    tts = StyleTTS2()
    reference_dicts = {}
    reference_dicts['696_92939'] = './Data/datasets/MEAD-Audio/M003/angry/level_3/001.m4a'
    start = time.time()
    noise = torch.randn(1, 1, 256).to(args.device)
    for k, path in reference_dicts.items():
        ref_s = tts.compute_style(path)
        wav = tts.inference(text, ref_s, alpha=0.3, beta=0.7, diffusion_steps=5, embedding_scale=1)
        rtf = (time.time() - start) / (len(wav) / 24000)
        print(f"RTF = {rtf:5f}")
        scaled_data = np.int16(wav * 32767)
        wavfile.write('./results/M003_test2.wav', 24000, scaled_data)

if __name__ == "__main__":
    main()



