# Copyright 2016 Tom SF Haines

# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at

#   http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.

import re
import string

from block import Block



has_letters_re = re.compile('[a-zA-Z]')
has_lower_re = re.compile('[a-z]')



def has_letters(text):
  return has_letters_re.search(text)!=None


def has_lower(text):
  return has_lower_re.search(text)!=None



def skip_starts_with(lines, word):
  if isinstance(word, str):
    word = (word, )
  
  for line in lines:
    line = line.strip()
    keep = True
    
    for w in word:
      if line.startswith(w):
        keep = False
        break
        
    if keep:
      yield line



def trim_lines(lines, start, end):
  """Given an iterator over lines and a start and end line yields lines between those ranges, including start, excluding end. 1 based, to match my text editor:-P"""
  for i, line in enumerate(lines, 1):
    if i>=start and i<end:
      yield line



def each_line(lines, attribution):
  """Given a generator from which it can yield lines this yields a Block for each line."""
  for line in lines:
    line = line.strip()
    if has_letters(line):
      yield Block(line, attribution)



def each_paragraph(lines, attribution):
  """Returns a block for each paragraph, where a paragraph is defined as being followed by a blank line."""
  buf = []
  for line in lines:
    line = line.strip()
    line = line.replace('_','')
    
    if has_letters(line):
      buf.append(line)
      
    elif len(buf)!=0:
      text = '\n'.join(buf)
      if has_lower(text):
        yield Block(text, attribution)
      buf = []
  
  if len(buf)!=0:
    text = '\n'.join(buf)
    if has_lower(text):
      yield Block(text, attribution)



def each_numbered_paragraph(lines, attribution):
  """Returns a block for each paragraph, where the first line of each paragraph contains a number. This number is removed but added to the attribution, which must contain a %i"""
  buf = []
  number = -1
  for line in lines:
    line = line.strip()
    line = line.replace('_','')
    
    if has_letters(line):
      buf.append(line)
      
    elif line.isdigit():
      number = int(line)
      
    elif len(buf)!=0:
      text = '\n'.join(buf)
      if has_lower(text):
        yield Block(text, attribution%number)
      buf = []
  
  if len(buf)!=0:
    text = '\n'.join(buf)
    if has_lower(text):
      yield Block(text, attribution%number)



def each_named_paragraph(lines, attribution):
  """Returns a block for each paragraph, where any block with no lowercase letters is converted into a title, and stuck into the attribution. (but not emitted as a block.)"""
  buf = []
  title = 'Unknown'
  for line in lines:
    line = line.strip()
    line = line.replace('_','')
    
    if has_letters(line): buf.append(line)
    elif len(buf)!=0:
      text = '\n'.join(buf)
      if has_lower(text):
        yield Block(text, attribution%title)
        
      elif len(text.replace('I','').replace('V','').replace('X',''))!=0:
        title = string.capwords(text.replace('\n',''))
      buf = []
  
  if len(buf)!=0:
    text = '\n'.join(buf)
    if has_lower(text):
      yield Block(text, attribution%title)



def each_prefixed_paragraph(lines, attribution, prefix):
  """Same as each_named_paragraph, but where each title has a prefix."""
  buf = []
  title = 'Unknown'
  for line in lines:
    line = line.strip()
    line = line.replace('_','')
    
    if has_letters(line):
      buf.append(line)
      
    elif len(buf)!=0:
      text = '\n'.join(buf)
      if has_lower(text):
        yield Block(text, attribution%title)
        
      elif text.startswith(prefix):
        title = string.capwords(text[len(prefix):].replace('\n',''))
      buf = []
  
  if len(buf)!=0:
    text = '\n'.join(buf)
    if has_lower(text):
      yield Block(text, attribution%title)



def each_named_indent(lines, attribution):
  """Returns a block for each paragraph, where any block that is not indented at all is converted into a title, and stuck into the attribution. (but not emitted as a block.)"""
  buf = []
  title = 'Unknown'
  for line in lines:
    line = line.replace('_','')
    
    if has_letters(line):
      buf.append(line)
      
    elif len(buf)!=0:
      text = '\n'.join(map(lambda s: s.strip(), buf))
      if has_lower(text):
        if buf[0][0]==' ':
          yield Block(text, attribution%title)
          
        else:
          title = text.replace('\n','')
      buf = []
  
  if len(buf)!=0:
    text = '\n'.join(buf)
    if has_lower(text):
      yield Block(text, attribution%title)



def skip_angled_blocks(lines):
  """Skips blocks of text contained within << and >>."""
  skipping = False
  for line in lines:
    l = line.strip()
    if skipping:
      if l.endswith('>>'):
        skipping = False
      
    else:
      if l.startswith('<<'):
        skipping = True
      else:
        yield line



def shakespeare(lines):
  """Custom parser for a certain well known bard."""
  play = None
  buf = []
  buf_indent = None
  
  for line in skip_angled_blocks(lines):
    l = line.strip()
    
    if has_letters(line):
      if play==None:
        if l=='by William Shakespeare':
          play = string.capwords(prev)
          act = None
          scene = None
        else: prev = line
        
      elif 'THE END' in l:
        play = None
        
      elif ('ACT' in l) and ('SCENE' in l):
        parts = l.replace('.',' ').split()
        act = parts[1]
        scene = parts[3]
        
      elif 'SCENE' in l:
        parts = l.replace('.',' ').split()
        if len(parts)>1: scene = parts[1]
        
      elif act!=None:
        indent = len(line)
        li = line.lstrip()
        indent -= len(li)
        
        if len(buf)==0:
          buf.append(l)
          buf_indent = indent
          
        elif indent>=buf_indent:
          buf.append(l)
          buf_indent = indent
          
        else:
          text = '\n'.join(buf)
          yield Block(text, '%s: Act %s, Scene %s - Shakespeare'%(play, act, scene))
          
          buf = []
          buf_indent = None
          
    elif len(buf)!=0:
      text = '\n'.join(buf)
      yield Block(text, '%s: Act %s, Scene %s - Shakespeare'%(play, act, scene))
          
      buf = []
      buf_indent = None
