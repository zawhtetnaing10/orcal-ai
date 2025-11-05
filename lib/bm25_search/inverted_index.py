from lib.utils.text_utils import tokenize_text


class InvertedIndex:
    def build(self):
        tokens = tokenize_text(
            "language skills I can speak Burmese fluently (obviously). I can speak English to some degree.")
        print(f"{tokens}")
