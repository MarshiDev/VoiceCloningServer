from chatterbox.tts import ChatterboxTTS
import threading
import torchaudio
import torch
import socket


queue: [(str, str, str)] = []     # text, voice_path, output_path
running = True

device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Using device: {device}")
model = ChatterboxTTS.from_pretrained(device=device)
conn: socket.socket


def tts_thread():
    while running:
        if queue:
            task = queue.pop(0)
            wav = model.generate(task[0], audio_prompt_path=task[1])
            torchaudio.save(task[2], wav, model.sr)
            conn.send(",".join(task).encode("utf-8"))


def main():
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.bind(("127.0.0.1", 9119))
    server.listen()

    threading.Thread(target=tts_thread()).start()

    global conn
    conn, _ = server.accept()

    while running:
        queue.append(conn.recv(1024).decode("utf-8").split(","))
