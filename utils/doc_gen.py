# Copyright (c) 2012, Tom SF Haines
# All rights reserved.

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#  * Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL <COPYRIGHT HOLDER> BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.



import pydoc



class DocGen:
  """A helper class that is used to generate documentation for the system. Outputs multiple formats simultaneously, specifically html for local reading with a webbrowser and the markup used by the wiki system on Google code."""
  def __init__(self, name, title = None, summary = ''):
    """name is the module name - primarilly used for the file names. title is the title used as applicable - if not provide it just uses the name. summary is an optional line to go below the title."""
    if title==None: title = name
    
    self.doc = pydoc.HTMLDoc()
    
    self.html = open('%s.html'%name,'w')
    self.html.write('<html>\n')
    self.html.write('<head>\n')
    self.html.write('<title>%s</title>\n'%title)
    self.html.write('</head>\n')
    self.html.write('<body>\n')
    
    self.variables = ''
    self.functions = ''
    self.classes = ''
    
  def __del__(self):
    if self.variables!='':
      self.html.write(self.doc.bigsection('Synonyms', '#ffffff', '#8d50ff', self.variables))
      
    if self.functions!='':
      self.html.write(self.doc.bigsection('Functions', '#ffffff', '#eeaa77', self.functions))
    
    if self.classes!='':
      self.html.write(self.doc.bigsection('Classes', '#ffffff', '#ee77aa', self.classes))
      
    self.html.write('</body>\n')
    self.html.write('</html>\n')
    self.html.close()
  
  
  def addFile(self, fn, title, fls = True):
    """Given a filename and section title adds the contents of said file to the output. Various flags influence how this works."""
    data = []
    for i, line in enumerate(open(fn,'r').readlines()):
      line = line.replace('\n', '')
      if i==0 and fls:
        line = '<strong>' + line + '</strong>'
      for ext in ['py','txt']:
        if '.%s - '%ext in line:
          s = line.split('.%s - '%ext, 1)
          line = '<i>' + s[0] + '.%s</i> - '%ext + s[1]
      data.append(line)
    self.html.write(self.doc.bigsection(title, '#ffffff', '#7799ee', '<br/>'.join(data)))
  
  
  def addVariable(self, var, desc):
    """Adds a variable to the documentation. Given the nature of this you provide it as a pair of strings - one referencing the variable, the other some kind of description of its use etc.."""
    self.variables += '<strong>%s</strong><br/>'%var
    self.variables += '%s<br/><br/>\n'%desc


  def addFunction(self, func):
    """Adds a function to the documentation. You provide the actual function instance."""
    self.functions += self.doc.docroutine(func).replace('&nbsp;',' ')
    self.functions += '\n'
  
  
  def addClass(self, cls):
    """Adds a class to the documentation. You provide the actual class object."""
    self.classes += self.doc.docclass(cls).replace('&nbsp;',' ')
    self.classes += '\n'
