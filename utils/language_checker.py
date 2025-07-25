from langdetect import detect


def detect_lang(text):
    detected_lang = detect(text)

    lang_mapping = {
        'en': 'english',
        'bn': 'bangla'
    }

    lang = lang_mapping.get(detected_lang, 'english')

    if detected_lang in ['bn', 'as', 'bpy', 'mni']:
        lang = 'bangla'

    return lang
