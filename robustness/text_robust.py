import numpy as np
import re
from tqdm import tqdm


##############################################################################
# Text
def text_robustness(tests, noise_level=0.3, swap=True, rand_mid=True, typo=True, sticky=True, omit=True):
    noises = []
    if swap:
        noises.append(swap_letter)
    if rand_mid:
        noises.append(random_mid)
    if typo:
        noises.append(qwerty_typo)
    if sticky:
        noises.append(sticky_keys)
    if omit:
        noises.append(omission)
    robustness_tests = []
    for i in tqdm(range(len(tests))):
        newtext = []
        text = normalizeText(tests[i])
        for word in text:
            if last_char(word) > 3 and np.random.sample() <= noise_level:
                mode = np.random.randint(len(noises))
                newtext.append(noises[mode](word))
            else:
                newtext.append(word)
        robustness_tests.append(' '.join(newtext))
    return robustness_tests


def normalizeText(text):
    text = text.lower()
    text = re.sub(r'<br />', r' ', text).strip()
    text = re.sub(r'^https?:\/\/.*[\r\n]*', ' L ', text, flags=re.MULTILINE)
    text = re.sub(r'[\~\*\+\^`_#\[\]|]', r' ', text).strip()
    text = re.sub(r'[0-9]+', r' N ', text).strip()
    text = re.sub(r'([/\'\-\.?!\(\)",:;])', r' \1 ', text).strip()
    return text.split()


def last_char(word):
    for i in range(len(word)):
        if word[len(word)-1-i].isalpha() or word[len(word)-1-i].isdigit():
            return len(word) - 1 - i
    return -1

def swap_letter(word):
    # swap two random adjacent letters
    last = last_char(word)
    pos = np.random.randint(last-2) + 1
    return word[:pos] + word[pos+1] + word[pos] + word[pos+2:]

def random_mid(word):
    # randomly permute the middle chunk of a word (all letters except the first and last letter)
    last = last_char(word)
    mid = [char for char in word[1:last]]
    np.random.shuffle(mid)
    return word[0]+''.join(mid)+word[last:]

def qwerty_typo(word):
    # randomly replace num_typo number of letters of a word to a one adjacent to it on qwerty keyboard
    qwerty = {'q':['w'], 'w':['q','e','s'], 'e':['w','r','d'], 'r':['e','t','f'], 't':['r','g','y'], 'y':['t','u','h'], 'u':['y','i','j'], 'i':['u','o','k'], 'o': ['i','p','l'], 'p':['o'], 'a': ['q','s','z'], 's':['a','w','d','x','z'], 'd':['s','e','f','x','c'], 'f':['d','r','g','c','v'], 'g':['f','t','h','v','b'], 'h':['g','y','j','b','n'], 'j':['h','u','k','n','m'], 'k':['j','i','l','m'], 'l':['k','o'], 'z':['a','s','x'], 'x':['z','s','d','c'], 'c':['x','d','f','v'], 'v':['c','f','g','b'], 'b':['v','g','h','n'], 'n':['b','h','m','j'], 'm':['n','j','k']}
    last = last_char(word)
    typos = np.arange(last+1)
    np.random.shuffle(typos)
    for i in range(len(typos)):
        if word[typos[i]] in qwerty:
            typo = qwerty[word[typos[i]]]
            key = typo[np.random.randint(len(typo))]
            word = word[:typos[i]] + key + word[typos[i]+1:]
            break
    return word

def sticky_keys(word, num_sticky=1):
    # randomly repeat num_sticky number of letters of a word
    last = last_char(word)
    sticky = np.arange(last+1)
    np.random.shuffle(sticky)
    for i in range(num_sticky):
        word = word[:sticky[i]] + word[sticky[i]] + word[sticky[i]:]
    return word

def omission(word, num_omit=1):
    # randomly omit num_omit number of letters of a word
    last = last_char(word)
    for i in range(num_omit):
        omit = np.random.randint(last-1) + 1
        word = word[:omit] + word[omit+1:]
        last -= 1
    return word