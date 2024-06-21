import os
import numpy as np
import torch
import yaml
from EAT.modules.generator import OcclusionAwareSPADEGeneratorEam
from EAT.modules.keypoint_detector import KPDetector, HEEstimator
import argparse
import imageio
from EAT.modules.transformer import Audio2kpTransformerBBoxQDeepPrompt as Audio2kpTransformer
from EAT.modules.prompt import EmotionDeepPrompt, EmotionalDeformationTransformer
from sync_batchnorm import DataParallelWithCallback
from scipy.io import wavfile

from EAT.modules.model_transformer import get_rotation_matrix, keypoint_transformation
from skimage import io, img_as_float32
from skimage.transform import resize
import torchaudio
import soundfile as sf
from scipy.spatial import ConvexHull
import cv2
import dlib

from skimage import transform as tf
import torch.nn.functional as F
import glob
from tqdm import tqdm
import gzip
import shutil

class EAT():
    def __init__(self, root_wav):
        self.root_wav = root_wav
        self.emo_label = ['ang',  'con',  'dis',  'fea',  'hap',  'neu',  'sad',  'sur']
        self.emo_label_full = ['angry',  'contempt',  'disgusted',  'fear',  'happy',  'neutral',  'sad',  'surprised']
        self.latent_dim = 16
        self.MEL_PARAMS_25 = {
            "n_mels": 80,
            "n_fft": 2048,
            "win_length": 640,
            "hop_length": 640
        }
        self.to_melspec = torchaudio.transforms.MelSpectrogram(**self.MEL_PARAMS_25)
        self.mean, self.std = -4, 4
        self.expU = torch.from_numpy(np.load('EAT/expPCAnorm_fin/U_mead.npy')[:,:32])
        self.expmean = torch.from_numpy(np.load('EAT/expPCAnorm_fin/mean_mead.npy'))

        # with open("config/vox-transformer2.yaml") as f:
        with open("EAT/config/deepprompt_eam3d_st_tanh_304_3090_all.yaml") as f:
            self.config = yaml.load(f, Loader=yaml.FullLoader)

        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor('EAT/demo/shape_predictor_68_face_landmarks.dat')

    @staticmethod
    def normalize_kp(kp_source, kp_driving, kp_driving_initial,use_relative_movement=True, use_relative_jacobian=True):
        kp_new = {k: v for k, v in kp_driving.items()}
        if use_relative_movement:
            kp_value_diff = (kp_driving['value'] - kp_driving_initial['value'])
            kp_new['value'] = kp_value_diff + kp_source['value']
            if use_relative_jacobian:
                jacobian_diff = torch.matmul(kp_driving['jacobian'], torch.inverse(kp_driving_initial['jacobian']))
                kp_new['jacobian'] = torch.matmul(jacobian_diff, kp_source['jacobian'])
        return kp_new

    @staticmethod
    def _load_tensor(data):
        wave_path = data
        wave, sr = sf.read(wave_path)
        wave_tensor = torch.from_numpy(wave).float()
        return wave_tensor

    def build_model(self):
        generator = OcclusionAwareSPADEGeneratorEam(**self.config['model_params']['generator_params'],
                                                     **self.config['model_params']['common_params'])
        if torch.cuda.is_available():
            print('cuda is available')
            generator.cuda()

        kp_detector = KPDetector(**self.config['model_params']['kp_detector_params'],
                                 **self.config['model_params']['common_params'])

        if torch.cuda.is_available():
            kp_detector.cuda()

        audio2kptransformer = Audio2kpTransformer(**self.config['model_params']['audio2kp_params'], face_ea=True)

        if torch.cuda.is_available():
            audio2kptransformer.cuda()

        sidetuning = EmotionalDeformationTransformer(**self.config['model_params']['audio2kp_params'])

        if torch.cuda.is_available():
            sidetuning.cuda()

        emotionprompt = EmotionDeepPrompt()

        if torch.cuda.is_available():
            emotionprompt.cuda()

        return generator, kp_detector, audio2kptransformer, sidetuning, emotionprompt


    def prepare_test_data(self, img_path, audio_path, opt, emotype, use_otherimg=True):
        # sr,_ = wavfile.read(audio_path)

        if use_otherimg:
            source_latent = np.load(img_path.replace('cropped', 'latent')[:-4]+'.npy', allow_pickle=True)
        else:
            source_latent = np.load(img_path.replace('images', 'latent')[:-9]+'.npy', allow_pickle=True)
        he_source = {}
        for k in source_latent[1].keys():
            he_source[k] = torch.from_numpy(source_latent[1][k][0]).unsqueeze(0).cuda()

        # source images
        source_img = img_as_float32(io.imread(img_path)).transpose((2, 0, 1))
        asp = os.path.basename(audio_path)[:-4]

        # latent code
        y_trg = self.emo_label.index(emotype)
        z_trg = torch.randn(self.latent_dim)

        # driving latent
        latent_path_driving = f'{self.root_wav}/latent_evp_25/{asp}.npy'
        # pose_gz = gzip.GzipFile(f'{self.root_wav}/poseimg/{asp}.npy.gz', 'r')
        pose_gz = gzip.GzipFile('EAT/demo/video_processed/template/poseimg/template.npy.gz','r')
        poseimg = np.load(pose_gz)
        deepfeature = np.load(f'{self.root_wav}/deepfeature32/{asp}.npy')
        driving_latent = np.load('EAT/demo/video_processed/template/latent_evp_25/template.npy', allow_pickle=True)
        he_driving = driving_latent[1]

        # gt frame number
        frames = glob.glob('EAT/demo/video_processed/template/images_evp_25/cropped/*.jpg')
        num_frames = len(frames)

        wave_tensor = self._load_tensor(audio_path)
        if len(wave_tensor.shape) > 1:
            wave_tensor = wave_tensor[:, 0]
        mel_tensor = self.to_melspec(wave_tensor)
        mel_tensor = (torch.log(1e-5 + mel_tensor) - self.mean) / self.std
        name_len = min(mel_tensor.shape[1], poseimg.shape[0], deepfeature.shape[0])

        audio_frames = []
        poseimgs = []
        deep_feature = []

        pad, deep_pad = np.load('EAT/pad.npy', allow_pickle=True)

        if name_len < num_frames:
            diff = num_frames - name_len
            if diff > 2:
                print(f"Attention: the frames are {diff} more than name_len, we will use name_len to replace num_frames")
                num_frames=name_len
                for k in he_driving.keys():
                    he_driving[k] = he_driving[k][:name_len, :]
        for rid in range(0, num_frames):
            audio = []
            poses = []
            deeps = []
            for i in range(rid - opt['num_w'], rid + opt['num_w'] + 1):
                if i < 0:
                    audio.append(pad)
                    poses.append(poseimg[0])
                    deeps.append(deep_pad)
                elif i >= name_len:
                    audio.append(pad)
                    poses.append(poseimg[-1])
                    deeps.append(deep_pad)
                else:
                    audio.append(mel_tensor[:, i])
                    poses.append(poseimg[i])
                    deeps.append(deepfeature[i])

            audio_frames.append(torch.stack(audio, dim=1))
            poseimgs.append(poses)
            deep_feature.append(deeps)
        audio_frames = torch.stack(audio_frames, dim=0)
        poseimgs = torch.from_numpy(np.array(poseimgs))
        deep_feature = torch.from_numpy(np.array(deep_feature)).to(torch.float)
        return audio_frames, poseimgs, deep_feature, source_img, he_source, he_driving, num_frames, y_trg, z_trg, latent_path_driving

    def load_ckpt(self, ckpt, kp_detector, generator, audio2kptransformer, sidetuning, emotionprompt):
        checkpoint = torch.load(ckpt, map_location=torch.device('cpu'))
        if audio2kptransformer is not None:
            audio2kptransformer.load_state_dict(checkpoint['audio2kptransformer'])
        if generator is not None:
            generator.load_state_dict(checkpoint['generator'])
        if kp_detector is not None:
            kp_detector.load_state_dict(checkpoint['kp_detector'])
        if sidetuning is not None:
            sidetuning.load_state_dict(checkpoint['sidetuning'])
        if emotionprompt is not None:
            emotionprompt.load_state_dict(checkpoint['emotionprompt'])



    # def shape_to_np(shape, dtype="int"):
    #     # initialize the list of (x, y)-coordinates
    #     coords = np.zeros((shape.num_parts, 2), dtype=dtype)

    #     # loop over all facial landmarks and convert them
    #     # to a 2-tuple of (x, y)-coordinates
    #     for i in range(0, shape.num_parts):
    #         coords[i] = (shape.part(i).x, shape.part(i).y)

    #     # return the list of (x, y)-coordinates
    #     return coords

    def crop_image(self, image_path, out_path):
        template = np.load('EAT/demo/M003_template.npy')
        image = cv2.imread(image_path)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        rects = self.detector(gray, 1)  #detect human face
        if len(rects) != 1:
            return 0
        for (j, rect) in enumerate(rects):
            shape = self.predictor(gray, rect) #detect 68 points
            coords = np.zeros((shape.num_parts, 2), dtype="int")
            for i in range(0, shape.num_parts):
                coords[i] = (shape.part(i).x, shape.part(i).y)
            shape = coords

        pts2 = np.float32(template[:47,:])
        pts1 = np.float32(shape[:47,:]) #eye and nose
        tform = tf.SimilarityTransform()
        tform.estimate( pts2, pts1) #Set the transformation matrix with the explicit parameters.

        dst = tf.warp(image, tform, output_shape=(256, 256))

        dst = np.array(dst * 255, dtype=np.uint8)

        cv2.imwrite(out_path, dst)

    def preprocess_imgs(self, allimgs, tmp_allimgs_cropped):
        name_cropped = []
        for path in tmp_allimgs_cropped:
            name_cropped.append(os.path.basename(path))
        for path in allimgs:
            if os.path.basename(path) in name_cropped:
                continue
            else:
                out_path = f'EAT/demo/imgs_cropped/{os.path.basename(path)}'
                self.crop_image(path, out_path)


    def load_checkpoints_extractor(self, config_path, checkpoint_path, cpu=False):

        with open(config_path) as f:
            config = yaml.load(f, Loader=yaml.FullLoader)

        kp_detector = KPDetector(**config['model_params']['kp_detector_params'],
                                 **config['model_params']['common_params'])
        if not cpu:
            kp_detector.cuda()

        he_estimator = HEEstimator(**config['model_params']['he_estimator_params'],
                                   **config['model_params']['common_params'])
        if not cpu:
            he_estimator.cuda()

        if cpu:
            checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
        else:
            checkpoint = torch.load(checkpoint_path)

        kp_detector.load_state_dict(checkpoint['kp_detector'])
        he_estimator.load_state_dict(checkpoint['he_estimator'])

        # if not cpu:
        #     kp_detector = DataParallelWithCallback(kp_detector)
        #     he_estimator = DataParallelWithCallback(he_estimator)

        kp_detector.eval()
        he_estimator.eval()

        return kp_detector, he_estimator

    @staticmethod
    def estimate_latent(driving_video, kp_detector, he_estimator):
        with torch.no_grad():
            predictions = []
            driving = torch.tensor(np.array(driving_video)[np.newaxis].astype(np.float32)).permute(0, 4, 1, 2, 3).cuda()
            kp_canonical = kp_detector(driving[:, :, 0])
            he_drivings = {'yaw': [], 'pitch': [], 'roll': [], 't': [], 'exp': []}

            for frame_idx in range(driving.shape[2]):
                driving_frame = driving[:, :, frame_idx]
                he_driving = he_estimator(driving_frame)
                for k in he_drivings.keys():
                    he_drivings[k].append(he_driving[k])
        return [kp_canonical, he_drivings]

    def extract_keypoints(self, extract_list):
        kp_detector, he_estimator = self.load_checkpoints_extractor(config_path='EAT/config/vox-256-spade.yaml', checkpoint_path='EAT/ckpt/pretrain_new_274.pth.tar')
        if not os.path.exists('EAT/demo/imgs_latent/'):
            os.makedirs('EAT/demo/imgs_latent/')
        for imgname in tqdm(extract_list):
            path_frames = [imgname]
            filesname=os.path.basename(imgname)[:-4]
            if os.path.exists(f'EAT/demo/imgs_latent/'+filesname+'.npy'):
                continue
            driving_frames = []
            for im in path_frames:
                driving_frames.append(imageio.imread(im))
            driving_video = [resize(frame, (256, 256))[..., :3] for frame in driving_frames]

            kc, he = self.estimate_latent(driving_video, kp_detector, he_estimator)
            kc = kc['value'].cpu().numpy()
            for k in he:
                he[k] = torch.cat(he[k]).cpu().numpy()
            np.save('EAT/demo/imgs_latent/'+filesname, [kc, he])

    def preprocess_cropped_imgs(self, allimgs_cropped):
        extract_list = []
        for img_path in allimgs_cropped:
            if not os.path.exists(img_path.replace('cropped', 'latent')[:-4]+'.npy'):
                extract_list.append(img_path)
        if len(extract_list) > 0:
            print('=========', "Extract latent keypoints from New image", '======')
            self.extract_keypoints(extract_list)

    def tf_generate(self, age, gender, emotype, save_dir=" "):
        # cur_path = os.getcwd()
        ckpt = 'EAT/ckpt/deepprompt_eam3d_all_final_313.pth.tar'
        generator, kp_detector, audio2kptransformer, sidetuning, emotionprompt = self.build_model()
        self.load_ckpt(ckpt, kp_detector=kp_detector, generator=generator, audio2kptransformer=audio2kptransformer, sidetuning=sidetuning, emotionprompt=emotionprompt)

        audio2kptransformer.eval()
        generator.eval()
        kp_detector.eval()
        sidetuning.eval()
        emotionprompt.eval()

        if gender == 'Male':
            allimg = glob.glob(f'EAT/demo/imgs/{age}/M/*.jpg')
        elif gender == 'Female':
            allimg = glob.glob(f'EAT/demo/imgs/{age}/W/*.jpg')
        allimg_name= [name.split('/')[-1] for name in allimg]
        all_wavs2 = glob.glob(f'{self.root_wav}/*.wav')

        # allimg = glob.glob('EAT/demo/imgs/*.jpg')
        tmp_allimg_cropped = glob.glob('EAT/demo/imgs_cropped/*.jpg')
        self.preprocess_imgs(allimg, tmp_allimg_cropped) # crop and align images

        allimg_cropped = glob.glob('EAT/demo/imgs_cropped/*.jpg')
        allcropped_name = ['EAT/demo/imgs_cropped/'+name for name in allimg_name]
        self.preprocess_cropped_imgs(allimg_cropped) # extract latent keypoints if necessary

        for ind in tqdm(range(len(all_wavs2))):
            for img_path in tqdm(allcropped_name):
                audio_path = all_wavs2[ind]
                # read in data
                audio_frames, poseimgs, deep_feature, source_img, he_source, he_driving, num_frames, y_trg, z_trg, latent_path_driving = \
                    self.prepare_test_data(img_path, audio_path, self.config['model_params']['audio2kp_params'], emotype)


                with torch.no_grad():
                    source_img = torch.from_numpy(source_img).unsqueeze(0).cuda()
                    kp_canonical = kp_detector(source_img, with_feature=True)     # {'value': value, 'jacobian': jacobian}
                    kp_cano = kp_canonical['value']

                    x = {}
                    x['mel'] = audio_frames.unsqueeze(1).unsqueeze(0).cuda()
                    x['z_trg'] = z_trg.unsqueeze(0).cuda()
                    x['y_trg'] = torch.tensor(y_trg, dtype=torch.long).cuda().reshape(1)
                    x['pose'] = poseimgs.cuda()
                    x['deep'] = deep_feature.cuda().unsqueeze(0)
                    x['he_driving'] = {'yaw': torch.from_numpy(he_driving['yaw']).cuda().unsqueeze(0),
                                    'pitch': torch.from_numpy(he_driving['pitch']).cuda().unsqueeze(0),
                                    'roll': torch.from_numpy(he_driving['roll']).cuda().unsqueeze(0),
                                    't': torch.from_numpy(he_driving['t']).cuda().unsqueeze(0),
                                    }

                    ### emotion prompt
                    emoprompt, deepprompt = emotionprompt(x)
                    a2kp_exps = []
                    emo_exps = []
                    T = 5
                    if T == 1:
                        for i in range(x['mel'].shape[1]):
                            xi = {}
                            xi['mel'] = x['mel'][:,i,:,:,:].unsqueeze(1)
                            xi['z_trg'] = x['z_trg']
                            xi['y_trg'] = x['y_trg']
                            xi['pose'] = x['pose'][i,:,:,:,:].unsqueeze(0)
                            xi['deep'] = x['deep'][:,i,:,:,:].unsqueeze(1)
                            xi['he_driving'] = {'yaw': x['he_driving']['yaw'][:,i,:].unsqueeze(0),
                                        'pitch': x['he_driving']['pitch'][:,i,:].unsqueeze(0),
                                        'roll': x['he_driving']['roll'][:,i,:].unsqueeze(0),
                                        't': x['he_driving']['t'][:,i,:].unsqueeze(0),
                                        }
                            he_driving_emo_xi, input_st_xi = audio2kptransformer(xi, kp_canonical, emoprompt=emoprompt, deepprompt=deepprompt, side=True)           # {'yaw': yaw, 'pitch': pitch, 'roll': roll, 't': t, 'exp': exp}
                            emo_exp = sidetuning(input_st_xi, emoprompt, deepprompt)
                            a2kp_exps.append(he_driving_emo_xi['emo'])
                            emo_exps.append(emo_exp)
                    elif T is not None:
                        for i in range(x['mel'].shape[1]//T+1):
                            if i*T >= x['mel'].shape[1]:
                                break
                            xi = {}
                            xi['mel'] = x['mel'][:,i*T:(i+1)*T,:,:,:]
                            xi['z_trg'] = x['z_trg']
                            xi['y_trg'] = x['y_trg']
                            xi['pose'] = x['pose'][i*T:(i+1)*T,:,:,:,:]
                            xi['deep'] = x['deep'][:,i*T:(i+1)*T,:,:,:]
                            xi['he_driving'] = {'yaw': x['he_driving']['yaw'][:,i*T:(i+1)*T,:],
                                        'pitch': x['he_driving']['pitch'][:,i*T:(i+1)*T,:],
                                        'roll': x['he_driving']['roll'][:,i*T:(i+1)*T,:],
                                        't': x['he_driving']['t'][:,i*T:(i+1)*T,:],
                                        }
                            he_driving_emo_xi, input_st_xi = audio2kptransformer(xi, kp_canonical, emoprompt=emoprompt, deepprompt=deepprompt, side=True)           # {'yaw': yaw, 'pitch': pitch, 'roll': roll, 't': t, 'exp': exp}
                            emo_exp = sidetuning(input_st_xi, emoprompt, deepprompt)
                            a2kp_exps.append(he_driving_emo_xi['emo'])
                            emo_exps.append(emo_exp)

                    if T is None:
                        he_driving_emo, input_st = audio2kptransformer(x, kp_canonical, emoprompt=emoprompt, deepprompt=deepprompt, side=True)           # {'yaw': yaw, 'pitch': pitch, 'roll': roll, 't': t, 'exp': exp}
                        emo_exps = sidetuning(input_st, emoprompt, deepprompt).reshape(-1, 45)
                    else:
                        he_driving_emo = {}
                        he_driving_emo['emo'] = torch.cat(a2kp_exps, dim=0)
                        emo_exps = torch.cat(emo_exps, dim=0).reshape(-1, 45)

                    exp = he_driving_emo['emo']
                    device = exp.get_device()
                    exp = torch.mm(exp, self.expU.t().to(device))
                    exp = exp + self.expmean.expand_as(exp).to(device)
                    exp = exp + emo_exps


                    source_area = ConvexHull(kp_cano[0].cpu().numpy()).volume
                    exp = exp * source_area

                    he_new_driving = {'yaw': torch.from_numpy(he_driving['yaw']).cuda(),
                                    'pitch': torch.from_numpy(he_driving['pitch']).cuda(),
                                    'roll': torch.from_numpy(he_driving['roll']).cuda(),
                                    't': torch.from_numpy(he_driving['t']).cuda(),
                                    'exp': exp}
                    he_driving['exp'] = torch.from_numpy(he_driving['exp']).cuda()

                    kp_source = keypoint_transformation(kp_canonical, he_source, False)
                    mean_source = torch.mean(kp_source['value'], dim=1)[0]
                    kp_driving = keypoint_transformation(kp_canonical, he_new_driving, False)
                    mean_driving = torch.mean(torch.mean(kp_driving['value'], dim=1), dim=0)
                    kp_driving['value'] = kp_driving['value']+(mean_source-mean_driving).unsqueeze(0).unsqueeze(0)
                    bs = kp_source['value'].shape[0]
                    predictions_gen = []
                    for i in tqdm(range(num_frames)):
                        kp_si = {}
                        kp_si['value'] = kp_source['value'][0].unsqueeze(0)
                        kp_di = {}
                        kp_di['value'] = kp_driving['value'][i].unsqueeze(0)
                        generated = generator(source_img, kp_source=kp_si, kp_driving=kp_di, prompt=emoprompt)
                        predictions_gen.append(
                            (np.transpose(generated['prediction'].data.cpu().numpy(), [0, 2, 3, 1])[0] * 255).astype(np.uint8))

                log_dir = save_dir
                os.makedirs(os.path.join(log_dir, "temp"), exist_ok=True)

                f_name = os.path.basename(img_path[:-4]) + "_" + emotype + "_" + os.path.basename(latent_path_driving)[:-4] + ".mp4"
                video_path = os.path.join(log_dir, "temp", f_name)
                # imageio.mimsave(video_path, predictions_gen, fps=25.0)
                imageio.mimsave(video_path, predictions_gen, format="MP4", fps=25)

                save_video = os.path.join(log_dir, f_name)
                cmd = r'ffmpeg -loglevel error -y -i "%s" -i "%s" -vcodec copy -shortest "%s"' % (video_path, audio_path, save_video)
                os.system(cmd)
                # os.remove(video_path)
                shutil.rmtree(os.path.join(log_dir, "temp"))

# if __name__ == '__main__':
#     argparser = argparse.ArgumentParser()
#     argparser.add_argument("--save_dir", type=str, default=" ", help="path of the output video")
#     argparser.add_argument("--name", type=str, default="deepprompt_eam3d_all_final_313", help="path of the output video")
#     argparser.add_argument("--emo", type=str, default="hap", help="emotion type ('ang',  'con',  'dis',  'fea',  'hap',  'neu',  'sad',  'sur')")
#     argparser.add_argument("--root_wav", type=str, default='./demo/video_processed/template')
#     args = argparser.parse_args()

#     root_wav=args.root_wav

#     if len(args.name) > 1:
#         name = args.name
#         print(name)
#     eat = EAT(args.root_wav)
#     eat.test(f'./ckpt/{name}.pth.tar', args.emo, save_dir=f'autodl-tmp/MERG/EAT/talking_head_testing/{name}/')
    
