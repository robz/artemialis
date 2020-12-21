import wave
import pygame

pygame.init()
clock = pygame.time.Clock()

# initialize audio
pygame.mixer.pre_init(frequency=44100, size=-16, channels=2)
pygame.mixer.init()


def play_sound(start, length, path):
    # open wave file
    wave_file = wave.open(path, 'rb')

    # skip unwanted frames
    n_frames = int(start * wave_file.getframerate())
    wave_file.setpos(n_frames)

    # write desired frames to audio buffer
    n_frames = int(length * wave_file.getframerate())
    frames = wave_file.readframes(n_frames)
    sound = pygame.mixer.Sound(buffer=frames)
    pygame.mixer.Sound.play(sound)

    # close and terminate everything properly
    wave_file.close()

    return sound

song = play_sound(
    start = 0,
    length = 30,
    path = 'Music/maestro-v2.0.0/2008/MIDI-Unprocessed_09_R1_2008_01-05_ORIG_MID--AUDIO_09_R1_2008_wav--3.wav',
)

pygame.time.wait(2000)
pygame.mixer.Sound.stop(song)

while True:
    clock.tick(60)
