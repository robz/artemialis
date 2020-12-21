import pygame
import sys
import wave

import pandas as pd
import matplotlib.pyplot as plt
import torch
from music21 import midi
import time
import random
from pygame import mixer
import threading
import torchaudio


start = time.time()
f = open("Documents/PerformanceMidi/human_training_time.txt", "r")
try:
    time_so_far = int(f.read())
except:
    time_so_far = 0
f.close()
print(F'spent {time_so_far} so far')



MAESTRO_DIRECTORY = 'Music/maestro-v2.0.0'

NUM_COMPOSERS = 5
CLIP_LEN = 30 # seconds

df = pd.read_csv(F'{MAESTRO_DIRECTORY}/maestro-v2.0.0.csv')
df['count'] = 1
df_deduped = df.groupby(by=['canonical_composer', 'canonical_title']).first()
df_deduped_train = df_deduped.where(df_deduped['split'] == 'train').dropna()
df_deduped_train.reset_index(inplace=True)

composers = list(df_deduped
  .groupby(by='canonical_composer')\
  .sum()\
  .sort_values('count', ascending=False)[:NUM_COMPOSERS].index)
composer_to_index = {composer: i for i, composer in enumerate(composers)}

df_deduped_train = df_deduped_train[df_deduped_train['canonical_composer'].isin(composers)]
df_deduped_train.reset_index(inplace=True)



path = MAESTRO_DIRECTORY + '/' + df_deduped_train['audio_filename'][0]
print(path)


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


def save_time():
    time_elapsed = time.time() - start
    total_time = int(time_elapsed + time_so_far)
    print(F'spent {time_elapsed} this time, in total {total_time}')
    f = open("Documents/PerformanceMidi/human_training_time.txt", "w")
    f.write(str(total_time))
    f.close()
    

def playsong():
    save_time()
    song_index = random.randint(0, len(df_deduped_train) - 1)
    path = MAESTRO_DIRECTORY + '/' + df_deduped_train['audio_filename'][song_index]
    length = pygame.mixer.Sound(path).get_length()
    start = random.randint(0, int(length) - CLIP_LEN - 1)
    song = play_sound(start, CLIP_LEN, path)
    return song_index, song


def main():
    pygame.init()
    clock = pygame.time.Clock()
    font_size = 24
    button_height = font_size + 10
    button_width = 300
    button_font = pygame.font.Font(pygame.font.get_default_font(), font_size)
    message_font = pygame.font.Font(pygame.font.get_default_font(), 12)

    fps = 60
    width = 450
    size = [width, 350]
    bg = [255, 255, 255]

    composer_text_surfaces = [
        button_font.render(c, True, (0, 0, 0)) for c in composers
    ]

    screen = pygame.display.set_mode(size)

    button_x = (width - button_width) / 2
    button_y = 70
    composer_buttons = [
        pygame.Rect(button_x, button_y + i * (button_height + 5), button_width, button_height) 
        for i in range(NUM_COMPOSERS)
    ]

    selected_button = None

    message = "Guess the composer!"
    score_message = "you can do it"
    submessage = None
    pause = False
    song_to_guess, song = playsong()

    corrects = 0
    total = 0

    while True:
        if pause == True:
            pygame.time.wait(5000)
            message = "Guess the composer!"
            selected_button = None
            submessage = None
            song_to_guess, song = playsong()
            pause = False

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False

            if event.type == pygame.MOUSEBUTTONDOWN:
                mouse_pos = event.pos
                for i, button in enumerate(composer_buttons):
                    if button.collidepoint(mouse_pos):
                        selected_button = i
            elif event.type == pygame.MOUSEBUTTONUP:
                mouse_pos = event.pos
                for i, button in enumerate(composer_buttons):
                    if button.collidepoint(mouse_pos):
                        mixer.Sound.stop(song) 

                        composer = df_deduped_train['canonical_composer'][song_to_guess]
                        title = df_deduped_train['canonical_title'][song_to_guess]
                        total += 1

                        if composer_to_index[composer] == i:
                            message = F'correct! {composer}'
                            corrects += 1
                        else:
                            message = F'wrong! {composer}'
                        submessage = F'{title}'
                        score_message = F'Win %: {int(corrects/total*100)}   ({corrects}/{total})'
                        print(F'{message} -- {submessage} -- {score_message}')

                        selected_button = i
                        pause = True


        screen.fill(bg)

        screen.blit(button_font.render(message, True, (0, 0, 0)), dest=(button_x, 10))

        if submessage is not None:
            screen.blit(message_font.render(submessage, True, (0, 0, 0)), dest=(button_x, 15 + font_size))

        for i, composer_text_surface in enumerate(composer_text_surfaces):
            color = [250, 250, 150] if selected_button == i else [150, 150, 150]
            pygame.draw.rect(screen, color, composer_buttons[i])
            screen.blit(composer_text_surface, dest=(button_x, button_y + i * (button_height + 5)))


        screen.blit(message_font.render(score_message, True, (0, 0, 0)), dest=(button_x, 320))

        pygame.display.update()
        clock.tick(fps)

    pygame.quit()
    sys.exit



if __name__ == '__main__':
    main()
