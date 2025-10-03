[Paxurux ✅ Chatter-box-dubbing](https://github.com/Paxurux/chatter-box-dubbing)


The owner of this open source project is the one at the link below:
https://github.com/Paxurux/chatter-box-dubbing







NZG Official - Stable Version of chatter-box-dubbing
This repository is a modified and stable copy of the
 original project, [Chatter-box-dubbing](https://github.com/Paxurux/chatter-box-dubbing)

🛑 Important Explanation: Reason for Re-uploading this Code


I, the owner of NZG Official, sincerely apologize to the original code owner, [Paxurux ](https://github.com/Paxurux)

 (( ✅ [Chatter-box-dubbing](https://github.com/Paxurux/chatter-box-dubbing) ✅), and wish to clarify that my intention is not in any way to infringe upon your copyright or claim ownership of your work.
I decided to download your repository and re-upload it to my account solely due to personal need and technical constraints:
 * PC Limitations: I do not have a powerful enough (PC 😭 😔) to run the code you release locally. My computer is very old and lacks a suitable graphics card.
 * Colab Usage: My primary need is to use this code in the ⚡[Google Colab ](https://colab.research.google.com/drive/1FZk_nxQZPGAWkGBWSl98AtQOTGCNoY56#scrollTo=C_Rr-Ritf5Wf) ⚡
    environment.
 * Updates Conflict: When you release a new update or feature in the original repository, that new code often conflicts or stops working with my existing Colab setup.
 * My Goal: Therefore, I uploaded a stable version here for personal use so I can maintain a version of the code that runs continuously in Colab without constant daily fixing.


🔗 Original Source and Credit

All credit and gratitude for the original project belong to its true creator, (✅ [Paxurux ](https://github.com/Paxurux) ✅)

 * ✅ Original Repository Link: https://github.com/Paxurux/chatter-box-dubbing
 * License: [ Please refer to the original repository's license]

I value your work and am using this copy only out of necessity.

✉📩📧 Contact and Assurance of Immediate Action
If you have any objection or legal concern regarding my upload of this repository, and you wish to issue a strike, please contact me before taking any action. I will immediately remove these files upon your request.

Contact Email: nzgnzg73@gmail.com

Thank you very much for your understanding and cooperation.



---![Screenshot 2025-06-04 132426](https://github.com/user-attachments/assets/6406d393-ee7d-4445-a5e9-92f07fd43e27)

![Screenshot 2025-06-04 132418](https://github.com/user-attachments/assets/7f423119-4901-406d-a48b-32c3d0bd891f)

````markdown
# 🎭 Chatterbox TTS Pro (SUP3R Edition)

**Chatterbox TTS Pro** is a high-quality, customizable text-to-speech (TTS) system enhanced with voice presets, advanced audio effects, conversation mode, and **AI video dubbing**. This is a fork of [Resemble AI's Chatterbox](https://github.com/resemble-ai/chatterbox), extended with additional audio controls, export options, and a powerful UI via Gradio.

---

## 🚀 Features

- 🎤 **Voice Presets**  
  Save and load voice settings including reference audio for fast reuse.

- 🎛️ **Advanced Audio Effects**  
  Add reverb, echo, pitch shifting, equalizer, 3D spatialization, and noise reduction.

- 🧠 **Conversation Mode**  
  Generate multi-speaker dialogues with different voice presets.

- 🎬 **AI Video Dubbing** ⭐ NEW!  
  Automatically dub videos into multiple languages with AI translation and voice synthesis.

- 📦 **Export Options**  
  Export audio in high/medium/low quality WAV formats.

- 🎚️ **Dynamic Controls**  
  Modify chunk size, temperature, seed, and more to fine-tune output.

---

## 🛠️ Installation

```bash
git clone https://github.com/SUP3RMASS1VE/chatterbox-SUP3R.git
cd chatterbox-SUP3R
python -m venv venv
venv\Scripts\activate
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
````

---

## ▶️ Usage

Run the app with:

```bash
python app.py
```

This will launch a Gradio-based interface in your browser where you can:

* **Text-to-Speech Tab**: Enter text and generate speech with advanced effects
* **Video Dubbing Tab**: Upload videos and automatically dub them into multiple languages
* Upload reference audio for voice cloning
* Enable advanced audio effects and 3D spatialization
* Save/load/delete voice presets
* Generate conversations between multiple speakers

## 🎬 Video Dubbing Setup

For the video dubbing feature, you'll need:

1. **Parakeet ASR Model**: 
   ```bash
   pip install nemo_toolkit[asr]
   ```

2. **Google Gemini API Key**: 
   - Get a free API key from [Google AI Studio](https://aistudio.google.com/app/apikey)
   - Add it in the "API Management" section of the Video Dubbing tab

3. **FFmpeg**: For video processing (usually included with most systems)

---

## 📁 File Structure

* `app.py` — Main application logic
* `voice_presets.json` — Stores saved voice preset metadata
* `saved_voices/` — Stores reference audio files
* `exports/` — Output directory for exported audio

---

## 💬 Conversation Mode Format

Use this format to define dialogues:

```
Alice: Hey Bob, how's it going?
Bob: Doing great! Just testing Chatterbox.
Alice: Awesome, it sounds incredible.
```

---

## 🙏 Acknowledgments

Big thanks to the original creators of Chatterbox:
👉 [Resemble AI](https://github.com/resemble-ai) for their groundbreaking work in controllable TTS.

---

## 📜 License

This project is provided under the original license of the upstream [chatterbox](https://github.com/resemble-ai/chatterbox). Check their repository for licensing details.

```

Let me know if you’d like a more developer-focused or user-focused version!
```
