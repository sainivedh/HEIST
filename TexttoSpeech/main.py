from gtts import gTTS
import os

def texttospeech(text, lang='hi'):

    output = gTTS(text=text, lang=lang, slow=True)

    output.save('output_audio.mp3')

    os.system('start output_audio.mp3')



#myText = "मंदई"

#language = 'hi'

#output = gTTS(text=myText, lang=language, slow=True)

#output.save('output_audio.mp3')

#os.system('start output_audio.mp3')
