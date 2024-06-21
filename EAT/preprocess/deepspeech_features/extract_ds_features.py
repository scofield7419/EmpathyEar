"""
    Script for extracting DeepSpeech features from audio file.
"""

import os
import argparse
import numpy as np
import pandas as pd
from EAT.preprocess.deepspeech_features.deepspeech_store import get_deepspeech_model_file
from EAT.preprocess.deepspeech_features.deepspeech_features import conv_audios_to_deepspeech



def extract_features(in_audios_path, out_files_path, deepspeech_pb_path='None'):
    """
    Real extract audio from video file.
    Parameters
    ----------
    in_audios : list of str
        Paths to input audio files.
    out_files : list of str
        Paths to output files with DeepSpeech features.
    deepspeech_pb_path : str
        Path to DeepSpeech 0.1.0 frozen model.
    metainfo_file_path : str, default None
        Path to file with meta-information.
    """
    if not os.path.exists(in_audios_path):
        raise Exception("Input file/directory doesn't exist: {}".format(in_audios_path))
    if deepspeech_pb_path is None:
        deepspeech_pb_path = ""
    if deepspeech_pb_path:
        deepspeech_pb_path = os.path.expanduser(deepspeech_pb_path)
    if not os.path.exists(deepspeech_pb_path):
        deepspeech_pb_path = get_deepspeech_model_file()

    audio_file_paths = []
    for file_name in os.listdir(in_audios_path):
        if not os.path.isfile(os.path.join(in_audios_path, file_name)):
            continue
        _, file_ext = os.path.splitext(file_name)
        if file_ext.lower() == ".wav":
            audio_file_path = os.path.join(in_audios_path, file_name) 
            audio_file_paths.append(audio_file_path)
    # l = len(audio_file_paths)
    # audio_file_paths = sorted(audio_file_paths)[2*l//3:]
    audio_file_paths = sorted(audio_file_paths)
    audio_file_paths_clean = []
    for i in audio_file_paths:
        if os.path.exists(i[:-4]+'.npy'):
            print('exists', i[:-4]+'.npy')
            continue
        else:
            audio_file_paths_clean.append(i)

    out_file_paths = [""] * len(audio_file_paths_clean)
    num_frames_info = [None] * len(audio_file_paths_clean)

    for i, in_audio in enumerate(audio_file_paths_clean):
        if not out_file_paths[i]:
            file_stem = os.path.splitext(in_audio)[0].split('/')[-1]
            out_file_paths[i] = out_files_path + '/' + file_stem + ".npy"
    conv_audios_to_deepspeech(
        audios=audio_file_paths_clean,
        out_files=out_file_paths,
        num_frames_info=num_frames_info,
        deepspeech_pb_path=deepspeech_pb_path)


# if __name__ == "__main__":
#     extract_features('TTS_results/test01','TTS_results/test01/deepfeature32')

