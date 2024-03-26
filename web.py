if __name__ == '__main__':
    import sys
    sys.path.append('./waveglow')
    print('loading')
from scipy.io import wavfile
from io import BytesIO
import gradio as gr
from text import symbols
from denoiser import Denoiser
from text import text_to_sequence
from train import load_model
from hparams import create_hparams
import numpy as np
import matplotlib.pyplot as plt
import torch




print('setting hparams')

hparams = create_hparams()
# hparams.sampling_rate = 22050
# hparams.max_decoder_steps = 250
hparams.gate_threshold = np.e**-4.5
hparams.p_attention_dropout = 0
hparams.p_decoder_dropout = 0
hparams.p_prenet_dropout = 0.5



print('loading checkpoint')
# checkpoint_path = "tacotron2_statedict.pt"
checkpoint_path = "output/checkpoint_44000"
model = load_model(hparams)
model.load_state_dict(torch.load(checkpoint_path)['state_dict'])
_ = model.cuda().eval()  # .half()



print('loading waveglow')
waveglow_path = 'waveglow_256channels_universal_v5_updated.pt'
waveglow = torch.load(waveglow_path)['model']

waveglow.cuda().eval()
for k in waveglow.convinv:
    k.float()
denoiser = Denoiser(waveglow)

def plot_data(data):
    for i in range(len(data)):
        plt.subplot(1, len(data), i+1)
        plt.imshow(data[i], aspect='auto', origin='lower',
                   interpolation='none')

def generate(text):
    sequence = np.array(text_to_sequence(text, ['chinese_cleaners']))[None, :]
    sequence = torch.autograd.Variable(
        torch.from_numpy(sequence)).cuda().long()
    sequence_text = ''.join([symbols[int(c)]
                             for c in sequence.float().data.cpu().numpy()[0]])

    mel_outputs, mel_outputs_postnet, gate_outputs, alignments = model.inference(
        sequence)

    plot_data((mel_outputs.float().data.cpu().numpy()[0],
               mel_outputs_postnet.float().data.cpu().numpy()[0],
               alignments.float().data.cpu().numpy()[0].T))
    plt.twinx()
    plt.plot(gate_outputs.float().data.cpu().numpy()[0, :, 0])

    with torch.no_grad():
        audio = waveglow.infer(
            mel_outputs_postnet.type(torch.float32), sigma=0.666)

    return audio[0].data.cpu().numpy(), sequence_text

def generate_audio(text):
    fig = plt.figure(figsize=(16, 4))
    audio_data, sequence_text = generate(text)
    # Convert to WAV in memory
    wavfile.write('./temp.wav', 22500, audio_data)
    return './temp.wav', fig, sequence_text


iface = gr.Interface(
    fn=generate_audio,
    inputs="text",
    outputs=["audio", "plot", "text"],
    allow_flagging=False)
iface.launch()
