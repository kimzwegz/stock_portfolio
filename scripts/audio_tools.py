from pydub import AudioSegment, silence
import math


class Split_audio(object):

    def __init__(self, sounds_path, filename):
        self.path = sounds_path
        self.filename = filename
        self.audio = AudioSegment.from_wav(self.path + "/" + self.filename)

    def cutbySilence(self, max_len=10, min_len=6, min_silence_len=500, r=1):
        # using dBFS to normalize the silence across files
        silence_thresh = self.audio.dBFS - 2 / r
        audio_splits = silence.split_on_silence(self.audio,
                                                min_silence_len=min_silence_len,
                                                # keep_silence=150,
                                                silence_thresh=silence_thresh)
        print(type(audio_splits))
        # cuts that are still too long, maybe an area of higher overall dBFS
        long_splits = [split for split in audio_splits if math.floor(split.duration_seconds) > max_len]
        if r != 2:
            for split in long_splits:
                audio_splits.remove(split)
                # cut recursively
                new_splits = self.cutbySilence(split, r=r + 1)
                for ns in new_splits:
                    audio_splits.append(ns)
        # clean the cuts of anything too short
        audio_splits = [split for split in audio_splits if math.floor(split.duration_seconds) > .5]
        # return audio_splits

        output_chunks = [audio_splits[0]]
        for chunk in audio_splits[1:]:
            if len(audio_splits[-1]) < min_len * 1000:
                audio_splits[-1] += chunk
            else:
                # if the last output chunk is longer than the target length,
                # we can start a new one
                audio_splits.append(chunk)
        return audio_splits