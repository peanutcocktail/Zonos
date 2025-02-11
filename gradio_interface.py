import torch
import torchaudio
import gradio as gr

from zonos.model import Zonos
from zonos.conditioning import make_cond_dict, supported_language_codes

if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"

CURRENT_MODEL_TYPE = None
CURRENT_MODEL = None


def load_model_if_needed(model_choice: str):
    global CURRENT_MODEL_TYPE, CURRENT_MODEL
    if CURRENT_MODEL_TYPE != model_choice:
        if CURRENT_MODEL is not None:
            del CURRENT_MODEL
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            elif torch.backends.mps.is_available():
                torch.mps.empty_cache()

        print(f"Loading {model_choice} model...")
        if model_choice == "Transformer":
            CURRENT_MODEL = Zonos.from_pretrained("Zyphra/Zonos-v0.1-transformer", device=device)
        else:
            CURRENT_MODEL = Zonos.from_pretrained("Zyphra/Zonos-v0.1-hybrid", device=device)
        CURRENT_MODEL.to(device)
        CURRENT_MODEL.bfloat16()
        CURRENT_MODEL.eval()
        CURRENT_MODEL_TYPE = model_choice
        print(f"{model_choice} model loaded successfully!")
    else:
        print(f"{model_choice} model is already loaded.")
    return CURRENT_MODEL


def update_ui(model_choice):
    """
    Dynamically show/hide UI elements based on the model's conditioners.
    We do NOT display 'language_id' or 'ctc_loss' even if they exist in the model.
    """
    model = load_model_if_needed(model_choice)
    cond_names = [c.name for c in model.prefix_conditioner.conditioners]
    print("Conditioners in this model:", cond_names)

    text_update = gr.update(visible=("espeak" in cond_names))
    language_update = gr.update(visible=("espeak" in cond_names))
    speaker_audio_update = gr.update(visible=("speaker" in cond_names))
    prefix_audio_update = gr.update(visible=True)
    skip_speaker_update = gr.update(visible=("speaker" in cond_names))
    skip_emotion_update = gr.update(visible=("emotion" in cond_names))
    emotion1_update = gr.update(visible=("emotion" in cond_names))
    emotion2_update = gr.update(visible=("emotion" in cond_names))
    emotion3_update = gr.update(visible=("emotion" in cond_names))
    emotion4_update = gr.update(visible=("emotion" in cond_names))
    emotion5_update = gr.update(visible=("emotion" in cond_names))
    emotion6_update = gr.update(visible=("emotion" in cond_names))
    emotion7_update = gr.update(visible=("emotion" in cond_names))
    emotion8_update = gr.update(visible=("emotion" in cond_names))
    skip_vqscore_8_update = gr.update(visible=("vqscore_8" in cond_names))
    vq_single_slider_update = gr.update(visible=("vqscore_8" in cond_names))
    fmax_slider_update = gr.update(visible=("fmax" in cond_names))
    skip_fmax_update = gr.update(visible=("fmax" in cond_names))
    pitch_std_slider_update = gr.update(visible=("pitch_std" in cond_names))
    skip_pitch_std_update = gr.update(visible=("pitch_std" in cond_names))
    speaking_rate_slider_update = gr.update(visible=("speaking_rate" in cond_names))
    skip_speaking_rate_update = gr.update(visible=("speaking_rate" in cond_names))
    dnsmos_slider_update = gr.update(visible=("dnsmos_ovrl" in cond_names))
    skip_dnsmos_ovrl_update = gr.update(visible=("dnsmos_ovrl" in cond_names))
    speaker_noised_checkbox_update = gr.update(visible=("speaker_noised" in cond_names))
    skip_speaker_noised_update = gr.update(visible=("speaker_noised" in cond_names))

    return (
        text_update,  # 1
        language_update,  # 2
        speaker_audio_update,  # 3
        prefix_audio_update,  # 4
        skip_speaker_update,  # 5
        skip_emotion_update,  # 6
        emotion1_update,  # 7
        emotion2_update,  # 8
        emotion3_update,  # 9
        emotion4_update,  # 10
        emotion5_update,  # 11
        emotion6_update,  # 12
        emotion7_update,  # 13
        emotion8_update,  # 14
        skip_vqscore_8_update,  # 15
        vq_single_slider_update,  # 16
        fmax_slider_update,  # 17
        skip_fmax_update,  # 18
        pitch_std_slider_update,  # 19
        skip_pitch_std_update,  # 20
        speaking_rate_slider_update,  # 21
        skip_speaking_rate_update,  # 22
        dnsmos_slider_update,  # 23
        skip_dnsmos_ovrl_update,  # 24
        speaker_noised_checkbox_update,  # 25
        skip_speaker_noised_update,  # 26
    )


def generate_audio(
    model_choice,
    text,
    language,
    speaker_audio,
    prefix_audio,
    skip_speaker,
    skip_emotion,
    e1,
    e2,
    e3,
    e4,
    e5,
    e6,
    e7,
    e8,
    skip_vqscore_8,
    vq_single,
    fmax,
    skip_fmax,
    pitch_std,
    skip_pitch_std,
    speaking_rate,
    skip_speaking_rate,
    dnsmos_ovrl,
    skip_dnsmos_ovrl,
    speaker_noised,
    skip_speaker_noised,
    cfg_scale,
    min_p,
    seed,
):
    """
    Generates audio based on the provided UI parameters.
    We do NOT use language_id or ctc_loss even if the model has them.
    """
    selected_model = load_model_if_needed(model_choice)

    uncond_keys = []
    if skip_speaker:
        uncond_keys.append("speaker")
    if skip_emotion:
        uncond_keys.append("emotion")
    if skip_vqscore_8:
        uncond_keys.append("vqscore_8")
    if skip_fmax:
        uncond_keys.append("fmax")
    if skip_pitch_std:
        uncond_keys.append("pitch_std")
    if skip_speaking_rate:
        uncond_keys.append("speaking_rate")
    if skip_dnsmos_ovrl:
        uncond_keys.append("dnsmos_ovrl")
    if skip_speaker_noised:
        uncond_keys.append("speaker_noised")

    speaker_noised_bool = bool(speaker_noised)
    fmax = float(fmax)
    pitch_std = float(pitch_std)
    speaking_rate = float(speaking_rate)
    dnsmos_ovrl = float(dnsmos_ovrl)
    cfg_scale = float(cfg_scale)
    min_p = float(min_p)
    seed = int(seed)
    max_new_tokens = 86 * 30

    torch.manual_seed(seed)

    speaker_embedding = None
    if speaker_audio is not None and not skip_speaker:
        wav, sr = torchaudio.load(speaker_audio)
        speaker_embedding = selected_model.make_speaker_embedding(wav, sr)
        speaker_embedding = speaker_embedding.to(device, dtype=torch.bfloat16)

    audio_prefix_codes = None
    if prefix_audio is not None:
        wav_prefix, sr_prefix = torchaudio.load(prefix_audio)
        wav_prefix = wav_prefix.mean(0, keepdim=True)
        wav_prefix = torchaudio.functional.resample(wav_prefix, sr_prefix, selected_model.autoencoder.sampling_rate)
        wav_prefix = wav_prefix.to(device, dtype=torch.float32)
        with torch.autocast(device, dtype=torch.float32):
            audio_prefix_codes = selected_model.autoencoder.encode(wav_prefix.unsqueeze(0))

    emotion_tensor = torch.tensor(
        [[float(e1), float(e2), float(e3), float(e4), float(e5), float(e6), float(e7), float(e8)]], device=device
    )

    vq_val = float(vq_single)
    vq_tensor = torch.tensor([vq_val] * 8, device=device).unsqueeze(0)

    cond_dict = make_cond_dict(
        text=text,
        language=language,
        speaker=speaker_embedding,
        emotion=emotion_tensor,
        vqscore_8=vq_tensor,
        fmax=fmax,
        pitch_std=pitch_std,
        speaking_rate=speaking_rate,
        dnsmos_ovrl=dnsmos_ovrl,
        speaker_noised=speaker_noised_bool,
        device=device,
        unconditional_keys=uncond_keys,
    )
    conditioning = selected_model.prepare_conditioning(cond_dict)

    codes = selected_model.generate(
        prefix_conditioning=conditioning,
        audio_prefix_codes=audio_prefix_codes,
        max_new_tokens=max_new_tokens,
        cfg_scale=cfg_scale,
        batch_size=1,
        sampling_params=dict(min_p=min_p),
    )

    wav_out = selected_model.autoencoder.decode(codes).cpu().detach()
    sr_out = selected_model.autoencoder.sampling_rate
    if wav_out.dim() == 2 and wav_out.size(0) > 1:
        wav_out = wav_out[0:1, :]
    return sr_out, wav_out.squeeze().numpy()


def build_interface():
    with gr.Blocks() as demo:
        with gr.Row():
            with gr.Column():
                model_choice = gr.Dropdown(
                    choices=["Hybrid", "Transformer"],
                    value="Transformer",
                    label="Zonos Model Type",
                    info="Select the model variant to use.",
                )
                text = gr.Textbox(
                    label="Text to Synthesize", value="Zonos uses eSpeak for text to phoneme conversion!", lines=4
                )
                language = gr.Dropdown(
                    choices=supported_language_codes,
                    value="en-us",
                    label="Language Code",
                    info="Select a language code.",
                )
            prefix_audio = gr.Audio(
                value="assets/silence_100ms.wav",
                label="Optional Prefix Audio (continue from this audio)",
                type="filepath",
            )
            with gr.Column():
                speaker_audio = gr.Audio(
                    label="Optional Speaker Audio (for cloning)",
                    type="filepath",
                )
                speaker_noised_checkbox = gr.Checkbox(label="Denoise Speaker?", value=False)

        with gr.Column():
            gr.Markdown("## Conditioning Parameters")

            with gr.Row():
                dnsmos_slider = gr.Slider(1.0, 5.0, value=4.0, step=0.1, label="DNSMOS Overall")
                fmax_slider = gr.Slider(0, 24000, value=22050, step=1, label="Fmax (Hz)")
                vq_single_slider = gr.Slider(0.5, 0.8, 0.78, 0.01, label="VQ Score")
                pitch_std_slider = gr.Slider(0.0, 400.0, value=20.0, step=1, label="Pitch Std")
                speaking_rate_slider = gr.Slider(0.0, 40.0, value=15.0, step=1, label="Speaking Rate")

            gr.Markdown("### Emotion Sliders")
            with gr.Row():
                emotion1 = gr.Slider(0.0, 1.0, 0.6, 0.05, label="Happiness")
                emotion2 = gr.Slider(0.0, 1.0, 0.05, 0.05, label="Sadness")
                emotion3 = gr.Slider(0.0, 1.0, 0.05, 0.05, label="Disgust")
                emotion4 = gr.Slider(0.0, 1.0, 0.05, 0.05, label="Fear")
            with gr.Row():
                emotion5 = gr.Slider(0.0, 1.0, 0.05, 0.05, label="Surprise")
                emotion6 = gr.Slider(0.0, 1.0, 0.05, 0.05, label="Anger")
                emotion7 = gr.Slider(0.0, 1.0, 0.5, 0.05, label="Other")
                emotion8 = gr.Slider(0.0, 1.0, 0.6, 0.05, label="Neutral")

            gr.Markdown("### Unconditional Toggles")
            with gr.Row():
                skip_speaker = gr.Checkbox(label="Skip Speaker", value=False)
                skip_emotion = gr.Checkbox(label="Skip Emotion", value=False)
                skip_vqscore_8 = gr.Checkbox(label="Skip VQ Score", value=True)
                skip_fmax = gr.Checkbox(label="Skip Fmax", value=False)
                skip_pitch_std = gr.Checkbox(label="Skip Pitch Std", value=False)
                skip_speaking_rate = gr.Checkbox(label="Skip Speaking Rate", value=False)
                skip_dnsmos_ovrl = gr.Checkbox(label="Skip DNSMOS", value=True)
                skip_speaker_noised = gr.Checkbox(label="Skip Noised Speaker", value=False)

        with gr.Column():
            gr.Markdown("## Generation Parameters")
            with gr.Row():
                cfg_scale_slider = gr.Slider(1.0, 5.0, 2.0, 0.1, label="CFG Scale")
                min_p_slider = gr.Slider(0.0, 1.0, 0.1, 0.01, label="Min P")
                seed_number = gr.Number(label="Seed", value=420, precision=0)

            generate_button = gr.Button("Generate Audio")
            output_audio = gr.Audio(label="Generated Audio", type="numpy")

        model_choice.change(
            fn=update_ui,
            inputs=[model_choice],
            outputs=[
                text,  # 1
                language,  # 2
                speaker_audio,  # 3
                prefix_audio,  # 4
                skip_speaker,  # 5
                skip_emotion,  # 6
                emotion1,  # 7
                emotion2,  # 8
                emotion3,  # 9
                emotion4,  # 10
                emotion5,  # 11
                emotion6,  # 12
                emotion7,  # 13
                emotion8,  # 14
                skip_vqscore_8,  # 15
                vq_single_slider,  # 16
                fmax_slider,  # 17
                skip_fmax,  # 18
                pitch_std_slider,  # 19
                skip_pitch_std,  # 20
                speaking_rate_slider,  # 21
                skip_speaking_rate,  # 22
                dnsmos_slider,  # 23
                skip_dnsmos_ovrl,  # 24
                speaker_noised_checkbox,  # 25
                skip_speaker_noised,  # 26
            ],
        )

        # On page load, trigger the same UI refresh
        demo.load(
            fn=update_ui,
            inputs=[model_choice],
            outputs=[
                text,
                language,
                speaker_audio,
                prefix_audio,
                skip_speaker,
                skip_emotion,
                emotion1,
                emotion2,
                emotion3,
                emotion4,
                emotion5,
                emotion6,
                emotion7,
                emotion8,
                skip_vqscore_8,
                vq_single_slider,
                fmax_slider,
                skip_fmax,
                pitch_std_slider,
                skip_pitch_std,
                speaking_rate_slider,
                skip_speaking_rate,
                dnsmos_slider,
                skip_dnsmos_ovrl,
                speaker_noised_checkbox,
                skip_speaker_noised,
            ],
        )

        # Generate audio on button click
        generate_button.click(
            fn=generate_audio,
            inputs=[
                model_choice,
                text,
                language,
                speaker_audio,
                prefix_audio,
                skip_speaker,
                skip_emotion,
                emotion1,
                emotion2,
                emotion3,
                emotion4,
                emotion5,
                emotion6,
                emotion7,
                emotion8,
                skip_vqscore_8,
                vq_single_slider,
                fmax_slider,
                skip_fmax,
                pitch_std_slider,
                skip_pitch_std,
                speaking_rate_slider,
                skip_speaking_rate,
                dnsmos_slider,
                skip_dnsmos_ovrl,
                speaker_noised_checkbox,
                skip_speaker_noised,
                cfg_scale_slider,
                min_p_slider,
                seed_number,
            ],
            outputs=[output_audio],
        )

    return demo


if __name__ == "__main__":
    demo = build_interface()
    demo.launch(server_name="0.0.0.0", server_port=7860, share=True)
