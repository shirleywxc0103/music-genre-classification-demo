import torch
torch.manual_seed(1234)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
import numpy as np
np.random.seed(0)

# from torch.optim.lr_scheduler import ReduceLROnPlateau
# import data_manager
# from src.cnn2D.model.custom_upchannel import *
from hparams import hparams
import torch.nn.functional as F
import librosa
import os


import torch
torch.manual_seed(123)
import torch.nn as nn
import torchvision
import torchaudio
# class upchannel(nn.Module):
#     def __init__(self, hparams):
#         super(upchannel, self).__init__()
#
#         self._extractor = nn.Sequential(
#             nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, stride=1, padding=1),
#             nn.BatchNorm2d(64),
#             nn.ReLU(),
#             nn.MaxPool2d(kernel_size=2),
#
#             nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
#             nn.BatchNorm2d(128),
#             nn.ReLU(),
#             nn.MaxPool2d(kernel_size=2),
#
#             nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),
#             nn.BatchNorm2d(256),
#             nn.ReLU(),
#             nn.MaxPool2d(kernel_size=4),
#
#             nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1),
#             nn.BatchNorm2d(512),
#             nn.ReLU(),
#             nn.MaxPool2d(kernel_size=4)
#         )
#
#         self._classifier = nn.Sequential(nn.Linear(in_features=2048, out_features=1024),
#                                          nn.ReLU(),
#                                          nn.Dropout(),
#                                          nn.Linear(in_features=1024, out_features=256),
#                                          nn.ReLU(),
#                                          nn.Dropout(),
#                                          nn.Linear(in_features=256, out_features=len(hparams.genres)))
#         self.apply(self._init_weights)
#
#     def forward(self, x):
#         x = torch.unsqueeze(x,1)
#         x = self._extractor(x)
#         x = x.view(x.size(0), -1)
#         score = self._classifier(x)
#         return score
#
#     def _init_weights(self, layer) -> None:
#         if isinstance(layer, nn.Conv1d):
#             nn.init.kaiming_uniform_(layer.weight)
#         elif isinstance(layer, nn.Linear):
#             nn.init.xavier_uniform_(layer.weight)
import torchvision.models as models

class upchannel(nn.Module):
    def __init__(self, hparams):
        super(upchannel, self).__init__()
        self.residual_block = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            # nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            # nn.MaxPool2d(kernel_size=4),

            # nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1),
            # nn.BatchNorm2d(512),
            # nn.ReLU(),
            # nn.MaxPool2d(kernel_size=4)
        )

        self.residual_block_2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            # nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            # nn.ReLU(),
            # nn.MaxPool2d(kernel_size=2),

            # nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            # nn.BatchNorm2d(64),
            # nn.ReLU(),
            # nn.MaxPool2d(kernel_size=4),

            # nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1),
            # nn.BatchNorm2d(512),
            # nn.ReLU(),
            # nn.MaxPool2d(kernel_size=4)
        )

        self.residual_block_3 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=3, kernel_size=1),
            # nn.BatchNorm2d(1),
            nn.ReLU(),

            nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3, stride=1, padding=1),
            # nn.BatchNorm2d(64),
            nn.ReLU(),
            # nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(in_channels=3, out_channels=1, kernel_size=1),
            # nn.BatchNorm2d(64),
            # nn.ReLU(),
            # nn.MaxPool2d(kernel_size=2),

            # nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            # nn.BatchNorm2d(64),
            # nn.ReLU(),
            # nn.MaxPool2d(kernel_size=4),

            # nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1),
            # nn.BatchNorm2d(512),
            # nn.ReLU(),
            # nn.MaxPool2d(kernel_size=4)
        )

        self.extend = nn.Conv2d(in_channels=1, out_channels=3, kernel_size=1)  # 扩展通道
        self.standard_resnet = models.resnet50(pretrained=True)  # 加载与训练模型resnet
        self.standard_resnet.fc = nn.Linear(2048, len(hparams.genres))
        self.relu = nn.ReLU()

        self._extractor = nn.Sequential(
            # nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, stride=1, padding=1),
            # nn.BatchNorm2d(64),
            # nn.ReLU(),
            # nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=4),

            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=4)
        )

        self._classifier = nn.Sequential(nn.Linear(in_features=2048, out_features=1024),
        # self._classifier = nn.Sequential(nn.Linear(in_features=512, out_features=1024),
                                         nn.ReLU(),
                                         nn.Dropout(),
                                         nn.Linear(in_features=1024, out_features=256),
                                         nn.ReLU(),
                                         nn.Dropout(),
                                         nn.Linear(in_features=256, out_features=len(hparams.genres)))

        self.conv_0 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=1)
        self.maxpooling = nn.MaxPool2d(kernel_size=2)
        self.avepooling = nn.AvgPool2d(kernel_size=2)

        self.apply(self._init_weights)

    def forward(self, x):
        # x = torch.unsqueeze(x,1)
        # x = self._extractor(x)
        # x = x.view(x.size(0), -1)
        # score = self._classifier(x)
        # return score

        # out = torch.unsqueeze(x, 1)

        # ################  my_resnet
        # x = torch.unsqueeze(x, 1)
        # out = self.residual_block(x)
        # x = self.conv_0(x)
        # x = self.maxpooling(x)
        # # print("x.shape:", x.shape)
        # # print("out.shape:", out.shape)
        # out = out + x
        # # print("out+x.shape:", out.shape)
        # out = self._extractor(out)
        # out = out.view(out.size(0), -1)
        # score = self._classifier(out)
        # return score
        # ################  my_resnet

        # ###### standard resnet
        # x = torch.unsqueeze(x, 1)
        # x = self.extend(x)
        x = self.standard_resnet(x)
        # print(x.shape)
        return x
        ###### standard resnet

        # ##### my_resnet_2 首先通道1->64，然后resnet(x) + x = out，然后extract，然后分类
        # x = torch.unsqueeze(x, 1)
        # x = self.conv_0(x)
        # out = self.residual_block_2(x)
        # out = out + x
        # out = self.relu(out)
        # # print("out.shape:", out.shape)
        # out1 = self.avepooling(out)
        # # print("out1.shape:", out1.shape)
        # out2 = self.avepooling(out)
        # # print("out2.shape:", out2.shape)
        # out = out1 + out2
        # # print("plus.shape:", out.shape)
        # # out = self.maxpooling(out)
        # out = self._extractor(out)
        # out = out.view(out.size(0), -1)
        # score = self._classifier(out)
        # return score
        # ##### my_resnet_2

        ##### my_resnet_3 XXXX首先通道1->64，然后resnet(x) + x = out，然后extract，然后分类
        # x = torch.unsqueeze(x, 1)
        # # print("x.shape:", x.shape)
        # out = self.residual_block_3(x)
        # # print("out.shape:", out.shape)
        # # out = self.residual_block_2(x)
        # out = out + x
        # out = self.relu(out)
        # out1 = self.avepooling(out)
        # # print("out1.shape:", out1.shape)
        # out2 = self.maxpooling(out)
        # # print("out2.shape:", out2.shape)
        # out = out1 + out2
        # # print("plus.shape:", out.shape)
        # # out = self.maxpooling(out)
        # out = self._extractor(out)
        # out = out.view(out.size(0), -1)
        # score = self._classifier(out)
        # return score
        ##### my_resnet_2

    def _init_weights(self, layer) -> None:
        if isinstance(layer, nn.Conv1d):
            nn.init.kaiming_uniform_(layer.weight)
        elif isinstance(layer, nn.Linear):
            nn.init.xavier_uniform_(layer.weight)
        elif isinstance(layer, nn.Conv2d):
            nn.init.kaiming_uniform_(layer.weight)


# Wrapper class to run PyTorch model
class Runner(object):
    def __init__(self, hparams):
        self.model = upchannel(hparams)
        if hparams.model_path != "":
            print(f"load model from {hparams.model_path}")
            self.model = torch.load(hparams.model_path, map_location="cpu")
        self.model.eval()
        print(f"model loading done")

        self.criterion = torch.nn.CrossEntropyLoss()
        # self.device = torch.device("cuda")
        self.device = torch.device("cpu")


        # if hparams.device > 0:
        #     torch.cuda.set_device(hparams.device - 1)
        #     self.model.cuda(hparams.device - 1)
        #     self.criterion.cuda(hparams.device - 1)
        #     self.device = torch.device("cuda:" + str(hparams.device - 1))

    # Accuracy function works like loss function in PyTorch
    def accuracy(self, source, target):
        source = source.max(1)[1].long().cpu()
        target = target.long().cpu()
        correct = (source == target).sum().item()

        return correct/float(source.size(0))

    def decode(self, data):
        self.model.eval()
        x = data.to(self.device)
        # print("x.shape:", x.shape)
        prediction = self.model(x)
        return prediction

    # Running model for train, test and validation. mode: 'train' for training, 'eval' for validation and test
    def run(self, dataloader, mode='train'):
        self.model.train() if mode is 'train' else self.model.eval()

        epoch_loss = 0
        epoch_acc = 0
        for batch, (x, y) in enumerate(dataloader):
            x = x.to(self.device)
            y = y.to(self.device)

            prediction = self.model(x)
            loss = self.criterion(prediction, y.long())
            acc = self.accuracy(prediction, y.long())

            if mode is 'train':
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()

            epoch_loss += prediction.size(0)*loss.item()
            epoch_acc += prediction.size(0)*acc

        epoch_loss = epoch_loss/len(dataloader.dataset)
        epoch_acc = epoch_acc/len(dataloader.dataset)

        return epoch_loss, epoch_acc, prediction

    # Early stopping function for given validation loss
    def early_stop(self, loss, epoch):
        self.scheduler.step(loss, epoch)
        self.learning_rate = self.optimizer.param_groups[0]['lr']
        stop = self.learning_rate < self.stopping_rate

        return stop

def melspectrogram(file_name, hparams):
    y, sr = librosa.load(file_name, hparams.sample_rate)
    S = librosa.stft(y, n_fft=hparams.fft_size, hop_length=hparams.hop_size, win_length=hparams.win_size)

    mel_basis = librosa.filters.mel(hparams.sample_rate, n_fft=hparams.fft_size, n_mels=hparams.num_mels)
    mel_S = np.dot(mel_basis, np.abs(S))
    mel_S = np.log10(1+10*mel_S)
    mel_S = mel_S.T

    return mel_S

def voting(rcs):
    #maority voting in pyth
    #
    # print(rcs)
    # print(type(rcs))
    # print("len(rcs):", len(rcs))
    max_label = max(rcs, key=rcs.count)
    print("max_label:", max_label)
    return max_label

def extract_features(audio_set):
    values = []
    for audio in audio_set:
        clip, sr = librosa.load(audio["name"], sr=22050)
        extract_spectrogram(values, clip, audio["class_idx"])
        print("Finished audio {}".format(audio))
    return values

def extract_spectrogram(values, clip, target):
    num_channels = 3
    window_sizes = [25, 50, 100]
    hop_sizes = [10, 25, 50]

    specs = []

    for i in range(num_channels):
        window_length = int(round(window_sizes[i] * 22050 / 1000))
        hop_length = int(round(hop_sizes[i] * 22050 / 1000))

        clip = torch.Tensor(clip)
        spec = torchaudio.transforms.MelSpectrogram(sample_rate=22050, n_fft=2205,
                                                    win_length=window_length, hop_length=hop_length, n_mels=128)(
            clip)  # Check this otherwise use 2400
        # print("spec.shape:", spec.shape)
        eps = 1e-6
        spec = spec.numpy()
        spec = np.log(spec + eps)
        # print("log(spec+eps):", spec.shape)
        spec = np.asarray(torchvision.transforms.Resize((128, 1500))(Image.fromarray(spec)))
        mean = 0.52550554
        std = 3.800251
        spec = (spec - mean) / std
        specs.append(spec)

    new_entry = {}
    new_entry["audio"] = clip.numpy()
    new_entry["values"] = np.array(specs)
    new_entry["target"] = target

    values.append(new_entry)


from datetime import datetime
def save_audio(file):
    # if file.size > 4000000:
    #     return 1
    # if not os.path.exists("audio"):
    #     os.makedirs("audio")
    folder = "demo_audio"
    datetoday = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
    # clear the folder to avoid storage overload
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))

    try:
        with open("log0.txt", "a") as f:
            f.write(f"{file.name} - {file.size} - {datetoday};\n")
    except:
        pass

    with open(os.path.join(folder, file.name), "wb") as f:
        f.write(file.getbuffer())
    return 0


import streamlit as st
import pandas as pd
import altair as alt
# import keras
from PIL import Image
import numpy as np

# from keras.utils.vis_utils import plot_model
# import seaborn as sns

# df = pd.read_csv('sales_data/vgsales.csv')
#option at the side bar
analysis = st.sidebar.selectbox('选择应用', ['音乐流派分类Demo'])
# analysis
#title
st.set_option('deprecation.showfileUploaderEncoding', False)
if analysis=='数据可视化分析':
    st.title('')
else:
    st.title('音乐流派分类Demo')
    st.write('本项目基于深度学习中的迁移学习算法，利用图像领域的预训练模型Resnet50，结合GTZAN数据集中音乐文件的梅尔谱图，对输入文件的音乐流派进行预测分析。')
    st.write('注：由于不同数据集分布的多样性，为保证分类准确率，请尽量使用GTZAN数据集中的音乐曲目进行试验。关于数据分布的问题，将在后续研究中继续进行...')
    st.write('所涉及音乐流派：蓝调，古典，乡村，迪斯科，嘻哈，爵士，金属，流行，雷鬼，摇滚。')
    st.subheader('请上传想要预测的音频文件。')
    wav_files = st.file_uploader("选择文件", type="wav", accept_multiple_files=True)
    print(wav_files)
    # st.subheader('Please upload a wav file of the target speaker.')
    # file_uploader_target = st.file_uploader("Select a wav file.", type="wav")
    # print(file_uploader_target)
    # model = keras.models.load_model('mnist_model', compile=False)
    if wav_files:
        for wav_file in wav_files:
            save_demo_audio = save_audio(wav_file)
            # path = '/home/xuechen/NKU/大四下/音乐流派分类/Code/20211208/dataset/gtzan/'
            path = 'demo_audio'
            genre = wav_file.name.split('.')[0]
            print(genre)
            # path += genre
            audio_file = open(path + '/' + wav_file.name, 'rb')
            audio_bytes = audio_file.read()
        # print(audio_bytes)
            a = st.audio(audio_bytes, format='audio/wav')

            # if st.button('开始分类'):
            #     st.write("预测分类中...")
            #     runner = Runner(hparams)
            #     file_name = wav_file
            #     hz_feat = melspectrogram(file_name, hparams)
            #     hz_feat = torch.from_numpy(hz_feat)
            #     print("hz_feat:", hz_feat)
            #     # hz_feat_2 = torch.randn(789, 128)
            #     print("hz_feat.shape:", hz_feat.shape)
            #     # print("hz_feat_2.shape:", hz_feat_2.shape)
            #     M = hz_feat.shape[0] // 128
            #     print("M:", M)
            #     rcs = []
            #     # test_loader, mean, std = data_manager.get_eval_dataloader(hparams)
            #     mean = 0.18094029
            #     std = 0.22892672
            #     print("mean:", mean)
            #     print("std:", std)
            #     for i in range(M):  # enumerate each 128*128 along axis-0
            #         data = hz_feat[i * 128:(i + 1) * 128, :]  # fetech each 128*128 chunk
            #         data = (data - mean) / std
            #         data = data.unsqueeze(0)  # 128*128 -> 1*128*128
            #         rc = runner.decode(data)  # call model to do forward propgation
            #         rc = rc[0]  # batch_size*tag_size -> tag_size
            #         print(f"{rc}")
            #         rc = torch.argmax(rc)  # find the position with max probability
            #         print(f"{i}th chunk got recognition result = {rc}")
            #         rcs.append(rc.cpu().item())
            #     # 3> voting function(rcs)
            #     prediction_result = voting(rcs)
            #
            #
            #     print("prediction result:", prediction_result)
            #
            #     st.write("分类结果:", hparams.app_genres[prediction_result])
            #     st.write("音乐流派介绍:", hparams.app_genres_discription[prediction_result])


            if st.button('开始分类'):
                root = '/home/xuechen/NKU/Datasets/GTZAN/genres'
                st.write("预测分类中...")
                runner = Runner(hparams)
                file_name = wav_file
                # genre = file_name.name.split('/')[0]
                # train_audio.append({"name": os.path.join(root, file_name), "class_idx": hparams.genres.index(genre)})
                audio = []
                audio.append({"name": os.path.join(path + '/' + wav_file.name), "class_idx": hparams.genres.index(genre)})
                print("audio:", audio)
                audio_set = []
                audio_set.extend(audio)
                audio_values = extract_features(audio_set)
                print("audio_values:", audio_values)

                # mean = 0.52550554
                # std = 3.800251

                x = []
                y = []
                for item in audio_values:
                    x.append(item["values"])
                    y.append(item["target"])

                rc = runner.decode(torch.tensor(x))




                # decoder_values = []
                # values = audio_values.index("values")
                # print("values:", values)
                # values = (values - mean) / std
                # decoder_values.append(values)
                    # data = data.unsqueeze(0)  # 128*128 -> 1*128*128
                # rc = runner.decode(decoder_values)  # call model to do forward propgation
                print("rc:", rc)
                rc = rc[0]  # batch_size*tag_size -> tag_size
                print("rc[0]:", rc)
                rc = torch.argmax(rc)  # find the position with max probability
                print("argmax(rc):", rc)
                # print(f"{i}th chunk got recognition result = {rc}")
                # rcs.append(rc.cpu().item())
                # 3> voting function(rcs)
                # prediction_result = voting(rcs)
                prediction_result = rc



                # hz_feat = melspectrogram(file_name, hparams)
                # hz_feat = torch.from_numpy(hz_feat)
                # print("hz_feat:", hz_feat)
                # # hz_feat_2 = torch.randn(789, 128)
                # print("hz_feat.shape:", hz_feat.shape)
                # # print("hz_feat_2.shape:", hz_feat_2.shape)
                # M = hz_feat.shape[0] // 128
                # print("M:", M)
                # rcs = []
                # # test_loader, mean, std = data_manager.get_eval_dataloader(hparams)
                # mean = 0.18094029
                # std = 0.22892672
                # print("mean:", mean)
                # print("std:", std)
                # for i in range(M):  # enumerate each 128*128 along axis-0
                #     data = hz_feat[i * 128:(i + 1) * 128, :]  # fetech each 128*128 chunk
                #     data = (data - mean) / std
                #     data = data.unsqueeze(0)  # 128*128 -> 1*128*128
                #     rc = runner.decode(data)  # call model to do forward propgation
                #     rc = rc[0]  # batch_size*tag_size -> tag_size
                #     print(f"{rc}")
                #     rc = torch.argmax(rc)  # find the position with max probability
                #     print(f"{i}th chunk got recognition result = {rc}")
                #     rcs.append(rc.cpu().item())
                # # 3> voting function(rcs)
                # prediction_result = voting(rcs)


                # print("prediction result:", prediction_result)

                st.write("分类结果:", hparams.app_genres[prediction_result])
                st.write("音乐流派介绍:", hparams.app_genres_discription[prediction_result])

