from googletrans import Translator

def translate(text, source_language):
    translator = Translator()
    translation = translator.translate(text, src=source_language, dest='en')
    return translation.text

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()