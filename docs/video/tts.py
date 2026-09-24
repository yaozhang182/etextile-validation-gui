import asyncio, json, os, subprocess, sys
from pathlib import Path
ROOT = str(Path(__file__).resolve().parents[2])
BUILD = os.environ.get('VIDEO_BUILD', os.path.join(ROOT, 'docs', 'video', 'build'))
sys.path.insert(0, os.path.dirname(__file__))
import edge_tts, imageio_ffmpeg
from script import BEATS
VOICE = 'en-US-AndrewNeural'
OUT = os.path.join(BUILD, 'audio')
FF = imageio_ffmpeg.get_ffmpeg_exe()
os.makedirs(OUT, exist_ok=True)

async def one(i, text):
    mp3 = f'{OUT}/beat_{i:02d}.mp3'; wav = mp3[:-4] + '.wav'
    if not os.path.exists(wav):
        await edge_tts.Communicate(text, VOICE, rate='-4%').save(mp3)
        subprocess.run([FF, '-y', '-loglevel', 'error', '-i', mp3, '-ac', '1', '-ar', '48000', wav], check=True)
    return wav

async def main():
    sem = asyncio.Semaphore(4)
    async def run(i, b):
        async with sem:
            return await one(i, b['say'])
    wavs = await asyncio.gather(*[run(i, b) for i, b in enumerate(BEATS)])
    import wave
    durs = []
    for p in wavs:
        with wave.open(p) as f:
            durs.append(f.getnframes() / f.getframerate())
    json.dump(durs, open(f'{OUT}/durations.json', 'w'))
    print(f'{len(durs)} clips, {sum(durs):.1f} s of narration')
asyncio.run(main())
