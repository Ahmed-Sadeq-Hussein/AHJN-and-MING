import torchaudio

from tts_formatter import synthesize


class CosyVoiceGenerator:

    def __init__(self, cosyvoice):
        self.cosyvoice = cosyvoice

    def generate(
        self,
        plan,
        reference_audio,
        reference_text,
        output_path
    ):
        audio = synthesize(
            cosyvoice=self.cosyvoice,
            plan=plan,
            prompt_wav=reference_audio,
            prompt_text=reference_text
        )

        torchaudio.save(
            output_path,
            audio.cpu(),
            self.cosyvoice.sample_rate
        )

        return output_path