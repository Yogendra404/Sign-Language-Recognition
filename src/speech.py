import subprocess
import threading

def _speak(text):
    safe = text.replace("'", "")
    cmd = (
        "Add-Type -AssemblyName System.Speech; "
        f"(New-Object System.Speech.Synthesis.SpeechSynthesizer).Speak('{safe}')"
    )
    subprocess.run(["powershell", "-Command", cmd], creationflags=0x08000000)

def speak_async(text):
    threading.Thread(target=_speak, args=(text,), daemon=True).start()