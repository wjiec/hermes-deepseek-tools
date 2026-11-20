#!/bin/sh
# Regenerates the audio fixtures from scratch with ffmpeg only (no speech, no third-party
# recordings): 0.8 s tone bursts separated by 0.4 s of silence, so silence detection has
# real cut points. Usage: sh tests/fixtures/audio/generate.sh
set -eu
cd "$(dirname "$0")"
burst() { # $1 = duration in seconds -> raw tone/silence pattern on stdin of the next ffmpeg
  ffmpeg -v error -f lavfi -i "sine=frequency=440:sample_rate=48000:duration=$1" \
    -af "volume='if(lt(mod(t,1.2),0.8),0.5,0)':eval=frame" -f wav -
}
burst 2.5 | ffmpeg -v error -y -i - -ac 1 -ar 48000 -c:a libopus -b:a 24k voice.ogg
burst 2.5 | ffmpeg -v error -y -i - -ac 1 -ar 16000 -c:a pcm_s16le pcm16k.wav
burst 2.5 | ffmpeg -v error -y -i - -ac 2 -ar 44100 -c:a pcm_f32le float_stereo.wav
burst 2.5 | ffmpeg -v error -y -i - -ar 22050 -c:a aac -b:a 64k clip.m4a
burst 2.5 | ffmpeg -v error -y -i - -ar 22050 -c:a libmp3lame -b:a 64k clip.mp3
burst 150.5 | ffmpeg -v error -y -i - -ac 1 -ar 48000 -c:a libopus -b:a 16k long.ogg
