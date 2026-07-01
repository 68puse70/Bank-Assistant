from deep_translator import GoogleTranslator
from indic_assistant.utils.logger import logger

class Translator:
    """Handles text translation using deep-translator (Google Translate)."""

    def __init__(self):
        pass

    def translate_to_english(self, text: str, src_lang: str) -> str:
        """Translates text from source language to English."""
        if src_lang == "en":
            return text
        if not text.strip():
            return ""

        logger.info(f"Translating text from '{src_lang}' to 'en'...")
        try:
            translator = GoogleTranslator(source=src_lang, target="en")
            result = translator.translate(text)
            logger.info(f"Translated to: '{result}'")
            return result
        except Exception as e:
            logger.warning(f"Translation to English failed ({e}); returning original text.")
            return text

    def translate_from_english(self, text: str, target_lang: str) -> str:
        """Translates text from English to the target Indian language."""
        if target_lang == "en":
            return text
        if not text.strip():
            return ""

        logger.info(f"Translating text from 'en' to '{target_lang}'...")
        try:
            translator = GoogleTranslator(source="en", target=target_lang)
            result = translator.translate(text)
            logger.info(f"Translated to: '{result}'")
            return result
        except Exception as e:
            logger.warning(f"Translation from English failed ({e}); returning original text.")
            return text
