from googletrans import Translator

def translate(text, source_language):
    translator = Translator()
    translation = translator.translate(text, src=source_language, dest='en')
    return translation.text