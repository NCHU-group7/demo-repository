from spellchecker import SpellChecker

spell = SpellChecker()

def correct_sentence(sentence):
    words = sentence.split()
    corrected_words = [spell.correction(word) or word for word in words]
    return ' '.join(corrected_words)

input_sentence = input("Enter a sentence: ")
corrected_sentence = correct_sentence(input_sentence)
print("Corrected Sentence:", corrected_sentence)
