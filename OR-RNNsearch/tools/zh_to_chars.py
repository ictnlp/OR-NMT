# -*- coding: utf-8 -*-

import io
import re
import string
from zhon import hanzi

def splitKeyWord(str):
    regex = r"[\u4e00-\ufaff]|[0-9]+|[a-zA-Z]+\'*[a-z]*"
    matches = re.findall(regex, str, re.UNICODE)
    return matches

def group_words(s):

    regex = []

    # Match a whole word:
    regex += [ur'[A-Za-z]+']

    # Match a single CJK character:
    regex += [ur'[\u4e00-\ufaff]']

    # Match one of anything else, except for spaces:
    #regex += [ur'^\s']

    # Match the float
    regex += [ur'[-+]?\d*\.\d+|\d+']

    # Match chinese float
    ch_punc = hanzi.punctuation
    regex += [ur'[{}]'.format(ch_punc)]	# point .

    # Match the punctuation
    regex += [ur'[.]+']	# point .

    punc = string.punctuation
    punc = punc.replace('.', '')
    regex += [ur'[{}]'.format(punc)]

    regex = "|".join(regex)
    r = re.compile(regex)

    return r.findall(s)

import argparse

parser = argparse.ArgumentParser(description='Chiese to characters.')
parser.add_argument('-i', '--input', dest='input', required=True, help='input file')
args = parser.parse_args()

input_file = args.input
lines = []
#with io.open('zh.1000', encoding='utf-8') as file:
with io.open(input_file, encoding='utf-8') as file:
    for line in file:
	lines.append(' '.join(group_words(line.strip())))

import codecs
#codecs.open('10000.chars', 'w', encoding='utf-8').write('\n'.join(lines) + '\n')
codecs.open('{}.char'.format(input_file), 'w', encoding='utf-8').write('\n'.join(lines) + '\n')



