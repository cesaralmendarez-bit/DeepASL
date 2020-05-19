#!/usr/bin/env python

from  AppKit import NSSpeechSynthesizer
import time
import sys
class sayLetter():

    def saySomething(letter):
        letterReal = 'G'

        if letter == 0:
            letterReal = 'A'
        elif letter == 1:
            letterReal = 'L'
        elif letter == 2:
            letterReal = 'B'
        elif letter == 3:
            letterReal = 'C'
        elif letter == 4:
            letterReal = 'H'

        text = "I think the letter is, {}".format(letterReal)

        nssp = NSSpeechSynthesizer

        ve = nssp.alloc().init()

        voice = nssp.availableVoices()[17]

        ve.setVoice_(voice)
        ve.startSpeakingString_(text)

        while not ve.isSpeaking():
            time.sleep(0.1)

        while ve.isSpeaking():
            time.sleep(0.1)
