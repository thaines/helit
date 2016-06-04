#! /usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright 2016 Tom SF Haines

# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at

#   http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.

import os.path
from collections import OrderedDict
import numpy

from extract_text import *
from ply2 import ply2



base = 'data'



# Do each item in turn...
db = []



## The vast majority of books can be processed in the same way - first create a list of these books and then process them in a loop...
simple = [('2 B R 0 2 B - Kurt Vonnegut.txt', 64, 443, None),
          ('A Christmas Carol - Charles Dickens.txt', 63, 3871, None),
          ('A Doll\'s House (play) - Henrik Ibsen.txt', 37, 3959, None),
          ('Alice\'s Adventures in Wonderland - Lewis Carroll.txt', 40, 3368, 'CHAPTER'),
          ('A Modest Proposal - Jonathan Swift.txt', 46, 367, None),
          ('An Inquiry into the Nature and Causes of the Wealth of Nations - Adam Smith.txt', 40, 35234, None),
          ('Anna Karenina - Leo Tolstoy.txt', 46, 43223, ('Part', 'Chapter')),
          ('Anne of Green Gables - L. M. Montgomery.txt', 81, 10816, 'CHAPTER'),
          ('Apology - Plato.txt', 43, 1427, None),
          ('A Princess of Mars - Edgar Rice Burroughs.txt', 235, 7175, None),
          ('A Study in Scarlet - Sir Arthur Conan Doyle.txt', 77, 4712, None),
          ('A Tale of Two Cities - Charles Dickens.txt', 105, 15900, 'Book the'),
          ('Candide - Voltaire.txt', 304, 4141, None),
          ('Contrasted Songs - Marian Longfellow.txt', 250, 2816, None),
          ('Crime and Punishment - Fyodor Dostoyevsky.txt', 127, 22091, None),
          ('David Copperfield - Charles Dickens.txt', 190, 38221, None),
          ('Don Juan - Lord Byron.txt', 49, 18229, '['),
          ('Dracula - Bram Stoker.txt', 91, 16269, None),
          ('Dubliners - James Joyce.txt', 59, 7844, None),
          ('Electric Gas Lighting - Norman H. Schneider.txt', 233, 1744, '[Illustration'),
          ('Emma - Jane Austen.txt', 37, 16261, None),
          ('Euthyphro - Plato.txt', 42, 1143, None),
          ('Frankenstein - Mary Wollstonecraft (Godwin) Shelley.txt', 48, 7279, ('Letter','Chapter')),
          ('Great Expectations - Charles Dickens.txt', 44, 20050, 'Chapter'),
          ('Gulliver\'s Travels - Jonathan Swift.txt', 106, 9324, None),
          ('Heart of Darkness - Joseph Conrad.txt', 38, 3336, None),
          ('Jane Eyre - Charlotte Brontë.txt', 186, 20701, None),
          ('Les Miserables - Victor Hugo.txt', 625, 67465, None),
          ('Madame Bovary - Gustave Flaubert.txt', 53, 13490, ('Chapter', 'Part')),
          ('Mari, Our Little Norwegian Cousin - Mary Hazelton Blanchard Wade.txt', 235, 2092, None),
          ('Metamorphosis - Franz Kafka.txt', 44, 1993, None),
          ('Moby Dick, or, the whale - Herman Melville.txt', 536, 21712, 'CHAPTER'),
          ('Myths and Legends of China - E. T. C. Werner.txt', 145, 12564, None),
          ('Narrative of the Life of Frederick Douglass - Frederick Douglass.txt', 76, 3675, None),
          ('Oliver Twist - Charles Dickens.txt', 151, 18832, None),
          ('Peter Pan - J. M. Barrie.txt', 84, 6242, 'Chapter'),
          ('Pride and Prejudice - Jane Austen.txt', 37, 13061, 'Chapter'),
          ('Raggedy Ann Stories - Johnny Gruelle.txt', 104, 2301, '[Illustration]'),
          ('Robinson Crusoe - Daniel Defoe.txt', 62, 9907, None),
          ('Secret Adversary - Agatha Christie.txt', 80, 10939, None),
          ('Sense and Sensibility - Jane Austen.txt', 43, 12653, None),
          ('Siddhartha - Hermann Hesse.txt', 50, 3954, None),
          ('The Adventures of Tom Sawyer - Mark Twain.txt', 457, 8859, None),
          ('The Boy Scouts at the Panama Canal - John Henry Goldfrap.txt', 118, 5888, None),
          ('The Brothers Karamazov - Fyodor Dostoyevsky.txt', 176, 35958, ('Book', 'Chapter')),
          ('The Call of the Wild - Jack London.txt', 52, 3062, 'Chapter'),
          ('The Communist Manifesto - Friedrich Engels, Karl Marx.txt', 38, 1520, None),
          ('The Count of Monte Cristo - Alexandre Dumas.txt', 38, 54179, 'Chapter'),
          ('The Divine Comedy - Dante.txt', 107, 15587, ('Complete', 'Cantos')),
          ('The Iliad - Homer.txt', 1965, 23248, '[Illustration'),
          ('The Importance of Being Earnest - Oscar Wilde.txt', 99, 3132, None),
          ('The Jungle Book - Rudyard Kipling.txt', 55, 5437, None),
          ('The Jungle - Upton Sinclair.txt', 43, 13872, 'Chapter'),
          ('The Legend of Sleepy Hollow - Washington Irving.txt', 35, 1156, None),
          ('The Mysterious Affair at Styles - Agatha Christie.txt', 58, 8440, None),
          ('The Picture of Dorian Gray - Oscar Wilde.txt', 45, 8534, None),
          ('The Prince - Niccolò Machiavelli.txt', 71, 4698, None),
          ('The Republic - Plato.txt', 8523, 24325, None),
          ('The Return of Sherlock Holmes - Sir Arthur Conan Doyle.txt', 77, 12654, None),
          ('The Rime of the Ancient Mariner - Samuel Taylor Coleridge.txt', 38, 850, None),
          ('The Romance of Lust - Anonymous.txt', 53, 180089, None),
          ('The Spirit Lake Massacre - Thomas Teakle.txt', 209, 6776, None),
          ('The Tale of Peter Rabbit - Beatrix Potter.txt', 66, 247, '[Illustration]'),
          ('The Time Machine - H. G. Wells.txt', 36, 3240, None),
          ('The Tyranny of Tears - Charles Haddon Chambers.txt', 257, 6589, None),
          ('The War of the Worlds - H. G. Wells.txt', 35, 6505, None),
          ('The Wonderful Wizard of Oz - L. Frank Baum.txt', 105, 4755, None),
          ('The Works of Edgar Allan Poe, Volume 1 - Edgar Allan Poe.txt', 956, 8858, None),
          ('The Works of Edgar Allan Poe, Volume 2 - Edgar Allan Poe.txt', 68, 9234, None),
          ('The Yellow Wallpaper - Charlotte Perkins Gilman.txt', 34, 873, None),
          ('Three Men in a Boat - Jerome K. Jerome.txt', 95, 7322, '[Picture:'),
          ('Through the Looking-Glass - Lewis Carroll.txt', 42, 3939, 'CHAPTER'),
          ('Treasure Island - Robert Louis Stevenson.txt', 140, 7521, 'PART'),
          ('Ulysses - James Joyce.txt', 34, 32690, None),
          ('Uncle Tom\'s Cabin - Harriet Beecher Stowe.txt', 46, 20879, None),
          ('Walden - Henry David Thoreau.txt', 94, 10135, None),
          ('War and Peace - Leo Tolstoy.txt', 36, 64975, 'BOOK'),
          ('Wuthering Heights - Emily Brontë.txt', 40, 12125, None),
          ('Your Mind and How to Use It - William Walker Atkinson.txt', 163, 4995, None)]


for fn, begin, end, skip_word in simple:
  f = open(os.path.join(base, fn))
  tr = trim_lines(f.xreadlines(), begin, end)
  
  if skip_word!=None:
    tr = skip_starts_with(tr, skip_word)
  
  for block in each_paragraph(tr, fn[:-4]):
    db.append(block)
    
  f.close()



## More complex files that require special casing...

### Guy De Maupassant - need to add book titles, as a bit unfair to reference just the collection...
f = open(os.path.join(base, 'Complete Original Short Stories of Guy De Maupassant - Guy de Maupassant.txt'))

for block in each_named_paragraph(trim_lines(f.xreadlines(), 580, 59479), '%s - Guy de Maupassant'):
  db.append(block)
  
f.close()


### Some fairy tales...
f = open(os.path.join(base, 'Grimms\' Fairy Tales - The Brothers Grimm.txt'))

for block in each_named_paragraph(trim_lines(f.xreadlines(), 125, 9182), '%s - Grimms\' Fairy Tales - The Brothers Grimm'):
  db.append(block)
  
f.close()


### Music...
f = open(os.path.join(base, 'Leaves of Grass - Walt Whitman.txt'))

for block in each_named_indent(trim_lines(f.xreadlines(), 33, 17886), '%s - Leaves of Grass - Walt Whitman'):
  db.append(block)
  
f.close()


### Greek and Roman legends...
f = open(os.path.join(base, 'Myths and Legends of Ancient Greece and Rome - E.M. Berens.txt'))

for block in each_named_paragraph(skip_starts_with(trim_lines(f.xreadlines(), 271, 10568), '[Illustration]'), '%s - Myths and Legends of Ancient Greece and Rome - E.M. Berens'):
  db.append(block)
  
f.close()


### Shakespeare, in two passes - first to do the sonnets, second the plays...
f = open(os.path.join(base, 'The Complete Works of William Shakespeare - William Shakespeare.txt'))

for block in each_numbered_paragraph(trim_lines(f.xreadlines(), 191, 2806), 'Sonnet %i - William Shakespeare'):
  db.append(block)
  
f.close()

f = open(os.path.join(base, 'The Complete Works of William Shakespeare - William Shakespeare.txt'))

for block in shakespeare(trim_lines(f.xreadlines(), 2825, 124376)):
  db.append(block)
  
f.close()


### Sherlock Holmes...
f = open(os.path.join(base, 'The Adventures of Sherlock Holmes - Arthur Conan Doyle.txt'))

for block in each_prefixed_paragraph(trim_lines(f.xreadlines(), 57, 12682), '%s - The Adventures of Sherlock Holmes - Arthur Conan Doyle', 'ADVENTURE '):
  db.append(block)
  
f.close()


### More short stories...
f = open(os.path.join(base, 'The Awakening and Selected Short Stories - Kate Chopin.txt'))

for block in each_named_paragraph(trim_lines(f.xreadlines(), 58, 7456), '%s - Kate Chopin'):
  db.append(block)
  
f.close()



# Print some stats...
print 'Extracted %i blocks'%len(db)


sentence_count = sum(map(lambda b: b.sentences, db))
print '%i sentences, an average of %.3f per block'%(sentence_count, float(sentence_count)/len(db))


line_count = sum(map(lambda b: b.lines, db))
print '%i lines, an average of %.3f per block'%(line_count, float(line_count)/len(db))


word_count = sum(map(lambda b: b.words, db))
print '%i words, an average of %.3f per block, %.3f per line or %.3f per sentence'%(word_count, float(word_count)/len(db), float(word_count) / line_count, float(word_count) / sentence_count)


letter_count = sum(map(lambda b: b.letters, db))
print '%i letters, an average of %.3f per block, %.3f per line, %.3f per sentence or %.3f per word.'%(letter_count, float(letter_count)/len(db), float(letter_count) / line_count, float(letter_count) / sentence_count, float(letter_count) / word_count)


digit_count = sum(map(lambda b: b.digits, db))
print '%i digits, an average of %.3f per block'%(digit_count, float(digit_count)/len(db))



# Save it out...
## First do a simple version, so it can be browsed and checked...
f = open('corpus.txt', 'w')
for block in db:
  f.write(block.text)
  f.write('\n\n      %s\n\n\n\n'%block.attribution)

f.close()


## Now do a ply2 file (ply2 loader cares about string encodings, so conversion to proper unicode objects is necesary given the text files actually contain utf8, and it will error out if not told this.)...
f_text = numpy.array([unicode(block.text, 'utf8') for block in db], dtype=numpy.object)
f_attribution = numpy.array([unicode(block.attribution, 'utf8') for block in db], dtype=numpy.object)


data = ply2.create()
data['type'].append('corpus')
data['comment'][0] = 'A large corpus of English literature, broken up by paragraph. Contents is entirly public domain and free for any use.'

data['element']['document'] = OrderedDict()
data['element']['document']['text'] = f_text
data['element']['document']['attribution'] = f_attribution

ply2.write('corpus.ply2', data)
