import math
from music21 import midi



####################################################################
# Functions to translate from midi files to sequences and back
####################################################################

def readMidi(filepath):
  mf = midi.MidiFile()
  mf.open(filepath)
  mf.read()
  mf.close()
  return mf


def writeMidi(mf, filename = 'tempmidi.mid'):
  mf.open(filename, attrib='wb')
  mf.write()
  mf.close()
  return filename

"""
def playMidi(mf):
  filename = writeMidi(mf)
  !fluidsynth -ni font.sf2 $filename -F $filename\.wav -r 16000 > /dev/null
  display(Audio(filename + '.wav'))
"""

event_to_idx = {}
for i in range(128):
  event_to_idx['note-on-' + str(i)] = i
for i in range(128):
  event_to_idx['note-off-' + str(i)] = i + 128
for i in range(100):
  event_to_idx['time-shift-' + str(i + 1)] = i + 128 + 128
for i in range(32):
  event_to_idx['velocity-' + str(i)] = i + 128 + 128 + 100


idx_to_event = list(event_to_idx.keys())
NUM_CHANNELS = len(idx_to_event)

def midiToIdxs(mf):
  ticks_per_beat = mf.ticksPerQuarterNote
  # The maestro dataset uses the first track to store tempo data
  tempo_data = next(e for e in mf.tracks[0].events if e.type == midi.MetaEvents.SET_TEMPO).data
  # tempo data is stored at microseconds per beat (beat = quarter note)
  microsecs_per_beat = int.from_bytes(tempo_data, 'big')
  millis_per_tick = microsecs_per_beat / ticks_per_beat / 1e3

  idxs = []
  started = False
  previous_t = None
  is_pedal_down = False
  notes_to_turn_off = set()

  # The second track stores the actual performance
  for e in mf.tracks[1].events:
    if started and e.type == 'DeltaTime' and e.time > 0:
      # event times are stored as ticks, so convert to milliseconds
      millis = e.time * millis_per_tick

      # combine repeated delta time events
      t = millis + (0 if previous_t is None else previous_t)

      # we can only represent a max time of 1 second (1000 ms)
      # so we must split up times that are larger than that into separate events
      while t > 0:
        t_chunk = min(t, 1000)
        idx = event_to_idx['time-shift-' + str(math.ceil(t_chunk / 10))]
        if previous_t is None:
          idxs.append(idx)
        else:
          idxs[-1] = idx
          previous_t = None
        t -= t_chunk
      previous_t = t_chunk

    elif e.type == midi.ChannelVoiceMessages.NOTE_ON:
      if e.velocity == 0:
        if is_pedal_down:
          notes_to_turn_off.add(e.pitch)
        else:
          idxs.append(event_to_idx['note-off-' + str(e.pitch)])
          previous_t = None
      else:
        if e.pitch in notes_to_turn_off:
          idxs.append(event_to_idx['note-off-' + str(e.pitch)])
          notes_to_turn_off.remove(e.pitch)

        # midi supports 128 velocities, but our representation only allows 32
        idxs.append(event_to_idx['velocity-' + str(e.velocity // 4)])
        idxs.append(event_to_idx['note-on-' + str(e.pitch)])
        started = True
        previous_t = None

    elif e.type == midi.ChannelVoiceMessages.CONTROLLER_CHANGE and e.parameter1 == 64: # sustain pedal
      # pedal values greater than 64 mean the pedal is being held down,
      # otherwise it's up
      if is_pedal_down and e.parameter2 < 64:
        is_pedal_down = False
        for pitch in notes_to_turn_off:
          idxs.append(event_to_idx['note-off-' + str(pitch)])
        notes_to_turn_off = set()
        previous_t = None
      elif not is_pedal_down and e.parameter2 >= 64:
        is_pedal_down = True
        previous_t = None

  return idxs


def makeNote(track, pitch, velocity):
  e = midi.MidiEvent(track, type=midi.ChannelVoiceMessages.NOTE_ON, channel=1)
  e.pitch = int(pitch)
  e.velocity = int(velocity)
  return e


def idxsToMidi(idxs, verbose=False):
  mf = midi.MidiFile()
  mf.ticksPerQuarterNote = 1024
  
  # The maestro dataset uses the first track to store tempo data, and the second
  # track to store the actual performance. So follow that convention.
  tempo_track = midi.MidiTrack(0)
  track = midi.MidiTrack(1)
  mf.tracks = [tempo_track, track]

  tempo = midi.MidiEvent(tempo_track, type=midi.MetaEvents.SET_TEMPO)
  # temp.data is the number of microseconds per beat (per quarter note)
  # So to set ticks per millis = 1 (easy translation from time-shift values to ticks),
  # tempo.data must be 1e3 * 1024, since ticksPerQuarterNote is 1024 (see above)
  tempo.data = int(1e3 * 1024).to_bytes(3, 'big')

  end_of_track = midi.MidiEvent(tempo_track, type=midi.MetaEvents.END_OF_TRACK)
  end_of_track.data = ''
  tempo_track.events = [
    # there must always be a delta time before each event
    midi.DeltaTime(tempo_track, time=0),
    tempo,
    midi.DeltaTime(tempo_track, time=0),
    end_of_track
  ]

  track.events = [midi.DeltaTime(track, time=0)]

  
  current_velocity = 0
  notes_on = set()
  errors = {'is_on': 0, 'is_not_on': 0}

  for idx in idxs:
    if 0 <= idx < 128: # note-on
      pitch = idx
      if pitch in notes_on:
        if verbose:
          print(pitch, 'is already on')
        errors['is_on'] += 1
        continue
      if track.events[-1].type != 'DeltaTime':
        track.events.append(midi.DeltaTime(track, time=0))
      track.events.append(makeNote(track, pitch, current_velocity))
      notes_on.add(pitch)

    elif 128 <= idx < (128 + 128): # note-off
      pitch = idx - 128
      if pitch not in notes_on:
        if verbose:
          print(pitch, 'is not on')
        errors['is_not_on'] += 1
        continue
      if track.events[-1].type != 'DeltaTime':
        track.events.append(midi.DeltaTime(track, time=0))
      track.events.append(makeNote(track, pitch, 0))
      notes_on.remove(pitch)

    elif (128 + 128) <= idx < (128 + 128 + 100): # time-shift
      t = (1 + idx - (128 + 128)) * 10
      if track.events[-1].type == 'DeltaTime':
        # combine repeated delta times
        track.events[-1].time += t
      else:
        track.events.append(midi.DeltaTime(track, time=t))

    else: # velocity
      current_velocity = (idx - (128 + 128 + 100)) * 4

  if verbose:
    print('remaining notes left on:', notes_on)

  if track.events[-1].type != 'DeltaTime':
    track.events.append(midi.DeltaTime(track, time=0))
  track.events.append(end_of_track)

  return mf, errors
