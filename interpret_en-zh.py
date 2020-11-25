# This Python file uses the following encoding: utf-8
# -*- coding: utf-8 -*

from __future__ import division
import logging
import subprocess
import os
from en_zh import *

import pyaudio
from gtts import gTTS
from pygame import mixer
import tempfile
def TTSspeak(sentence, lang, loops=1):
    with tempfile.NamedTemporaryFile(delete=True) as fp:
        tts=gTTS(text=sentence, lang=lang)
        tts.save('{}.mp3'.format(fp.name))
        mixer.init()
        mixer.music.load('{}.mp3'.format(fp.name))
        mixer.music.play(loops)

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

from colorama import Fore, Back, Style 
'''
print(Fore.RED + 'some red text') 
print(Back.GREEN + 'and with a green background') 
print(Style.DIM + 'and in dim text') 
print(Style.RESET_ALL) 
print('back to normal now') 
'''
#====================================================

import time
import re
import sys
from google.cloud import speech
import pyaudio
from six.moves import queue

# Audio recording parameters
STREAMING_LIMIT = 55000
SAMPLE_RATE = 16000
CHUNK_SIZE = int(SAMPLE_RATE / 10)  # 100ms

def get_current_time():
    return int(round(time.time() * 1000))

def duration_to_secs(duration):
    return duration.seconds + (duration.nanos / float(1e9))

class ResumableMicrophoneStream:
    """Opens a recording stream as a generator yielding the audio chunks."""
    def __init__(self, rate, chunk_size):
        self._rate = rate
        self._chunk_size = chunk_size
        self._num_channels = 1
        self._max_replay_secs = 5

        # Create a thread-safe buffer of audio data
        self._buff = queue.Queue()
        self.closed = True
        self.start_time = get_current_time()

        # 2 bytes in 16 bit samples
        self._bytes_per_sample = 2 * self._num_channels
        self._bytes_per_second = self._rate * self._bytes_per_sample

        self._bytes_per_chunk = (self._chunk_size * self._bytes_per_sample)
        self._chunks_per_second = (
                self._bytes_per_second // self._bytes_per_chunk)

    def __enter__(self):
        self.closed = False

        self._audio_interface = pyaudio.PyAudio()
        self._audio_stream = self._audio_interface.open(
            format=pyaudio.paInt16,
            channels=self._num_channels,
            rate=self._rate,
            input=True,
            frames_per_buffer=self._chunk_size,
            # Run the audio stream asynchronously to fill the buffer object.
            # This is necessary so that the input device's buffer doesn't
            # overflow while the calling thread makes network requests, etc.
            stream_callback=self._fill_buffer,
        )

        return self

    def __exit__(self, type, value, traceback):
        self._audio_stream.stop_stream()
        self._audio_stream.close()
        self.closed = True
        # Signal the generator to terminate so that the client's
        # streaming_recognize method will not block the process termination.
        self._buff.put(None)
        self._audio_interface.terminate()

    def _fill_buffer(self, in_data, *args, **kwargs):
        """Continuously collect data from the audio stream, into the buffer."""
        self._buff.put(in_data)
        return None, pyaudio.paContinue

    def generator(self):
        while not self.closed:
            if get_current_time() - self.start_time > STREAMING_LIMIT:
                self.start_time = get_current_time()
                break
            # Use a blocking get() to ensure there's at least one chunk of
            # data, and stop iteration if the chunk is None, indicating the
            # end of the audio stream.
            chunk = self._buff.get()
            if chunk is None:
                return
            data = [chunk]

            # Now consume whatever other data's still buffered.
            while True:
                try:
                    chunk = self._buff.get(block=False)
                    if chunk is None:
                        return
                    data.append(chunk)
                except queue.Empty:
                    break

            yield b''.join(data)

def listen_print_loop(responses, stream):
    
    """Iterates through server responses and prints them.

    The responses passed is a generator that will block until a response
    is provided by the server.

    Each response may contain multiple results, and each result may contain
    multiple alternatives; for details, see https://goo.gl/tjCPAU.  Here we
    print only the transcription for the top alternative of the top result.

    In this case, responses are provided for interim results as well. If the
    response is an interim one, print a line feed at the end of it, to allow
    the next result to overwrite it, until the response is a final one. For the
    final one, print a newline to preserve the finalized transcription.
    """
    responses = (r for r in responses if (
            r.results and r.results[0].alternatives))

    num_chars_printed = 0
    for response in responses:
        if not response.results:
            continue

        # The `results` list is consecutive. For streaming, we only care about
        # the first result being considered, since once it's `is_final`, it
        # moves on to considering the next utterance.
        result = response.results[0]
        if not result.alternatives:
            continue

        # Display the transcription of the top alternative.
        top_alternative = result.alternatives[0]
        transcript = top_alternative.transcript # response message (text message)

        # Display interim results, but with a carriage return at the end of the
        # line, so subsequent lines will overwrite them.
        #
        # If the previous result was longer than this one, we need to print
        # some extra spaces to overwrite the previous result
        overwrite_chars = ' ' * (num_chars_printed - len(transcript))

        if not result.is_final:
            sys.stdout.write(transcript + overwrite_chars + '\r')

            num_chars_printed = len(transcript)

        else:
            
            predicted_seq, _ = translate(transcript)
            target_vocab_size = subword_encoder_zh.vocab_size
            predicted_seq_without_bos_eos = [idx for idx in predicted_seq if idx < target_vocab_size]
            trans_result = subword_encoder_zh.decode(predicted_seq_without_bos_eos)

            print(Fore.GREEN + Back.RED + '======== You speak: ========')
            print(transcript + overwrite_chars)
            print(Fore.BLUE + Back.YELLOW + '======== My translation: ========')
            print(trans_result)
            TTSspeak(trans_result, 'zh-tw')
            time.sleep(5)

            break

            #subprocess.call(["./tts_offline_sample", "-i", b, "-o", "tts_sample.wav"])
            #os.system("mplayer tts_sample.wav")                
            #if re.search(r'\b(exit|quit)\b', transcript, re.I):
            #    print('Exiting..')
            #    stream.closed = True
            #    break
            
            num_chars_printed = 0

def run_streaming():

    client = speech.SpeechClient().from_service_account_json('Your Project.json')
    config = speech.types.RecognitionConfig(
        encoding=speech.enums.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=SAMPLE_RATE,
        language_code='en-US', #English
        #language_code='zh-TW', #Chinese
		max_alternatives=1,
        enable_word_time_offsets=True)
    streaming_config = speech.types.StreamingRecognitionConfig(
        config=config,
        interim_results=True)

    mic_manager = ResumableMicrophoneStream(SAMPLE_RATE, CHUNK_SIZE)

    with mic_manager as stream:
        while not stream.closed:
            audio_generator = stream.generator()
            requests = (speech.types.StreamingRecognizeRequest(
                audio_content=content)
                for content in audio_generator)
 
            responses = client.streaming_recognize(streaming_config,
                                                   requests)            						   												   

            print('Please speak: ')
            listen_print_loop(responses, stream)

def _main():
    logging.basicConfig(level="INFO") #debug message    
    speak_text = "Hello, I can translate English into Chinese for you."
    print(speak_text)
    #TTSspeak(speak_text, 'en-us')
    #time.sleep(4)
    #subprocess.call(["./tts_offline_sample", "-i", speak_text, "-o", "tts_sample.wav"])
    #subprocess.call(["mplayer", "tts_sample.wav"])
    #os.system("mplayer tts_sample.wav")
    #time.sleep(5)
    run_streaming()

if __name__ == '__main__':
    _main()