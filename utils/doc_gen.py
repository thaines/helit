# Copyright (c) 2012, Tom SF Haines
# All rights reserved.

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#  * Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL <COPYRIGHT HOLDER> BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.



import pydoc
import inspect



class DocGen:
  """A helper class that is used to generate documentation for the system. Outputs multiple formats simultaneously, specifically html for local reading with a webbrowser and the markup used by the wiki system on Google code."""
  def __init__(self, name, title = None, summary = None):
    """name is the module name - primarilly used for the file names. title is the title used as applicable - if not provide it just uses the name. summary is an optional line to go below the title."""
    if title==None: title = name
    if summary==None: summary = title
    
    self.doc = pydoc.HTMLDoc()
    
    self.html = open('%s.html'%name,'w')
    self.html.write('<html>\n')
    self.html.write('<head>\n')
    self.html.write('<title>%s</title>\n'%title)
    self.html.write('</head>\n')
    self.html.write('<body>\n')
    
    self.html_variables = ''
    self.html_functions = ''
    self.html_classes = ''
    
    self.wiki = open('%s.wiki'%name,'w')
    self.wiki.write('#summary %s\n\n'%summary)
    self.wiki.write('= %s= \n\n'%title)
    
    self.wiki_variables = ''
    self.wiki_functions = ''
    self.wiki_classes = ''
    
  def __del__(self):
    if self.html_variables!='':
      self.html.write(self.doc.bigsection('Synonyms', '#ffffff', '#8d50ff', self.html_variables))
      
    if self.html_functions!='':
      self.html.write(self.doc.bigsection('Functions', '#ffffff', '#eeaa77', self.html_functions))
    
    if self.html_classes!='':
      self.html.write(self.doc.bigsection('Classes', '#ffffff', '#ee77aa', self.html_classes))
      
    self.html.write('</body>\n')
    self.html.write('</html>\n')
    self.html.close()
    
    
    if self.wiki_variables!='':
      self.wiki.write('= Variables =\n\n')
      self.wiki.write(self.wiki_variables)
      self.wiki.write('\n')
      
    if self.wiki_functions!='':
      self.wiki.write('= Functions =\n\n')
      self.wiki.write(self.wiki_functions)
      self.wiki.write('\n')

    if self.wiki_classes!='':
      self.wiki.write('= Classes =\n\n')
      self.wiki.write(self.wiki_classes)
      self.wiki.write('\n')

    self.wiki.close()
  
  
  def addFile(self, fn, title, fls = True):
    """Given a filename and section title adds the contents of said file to the output. Various flags influence how this works."""
    html = []
    wiki = []
    for i, line in enumerate(open(fn,'r').readlines()):
      hl = line.replace('\n', '')
      if i==0 and fls:
        hl = '<strong>' + hl + '</strong>'
      for ext in ['py','txt']:
        if '.%s - '%ext in hl:
          s = hl.split('.%s - '%ext, 1)
          hl = '<i>' + s[0] + '.%s</i> - '%ext + s[1]
      html.append(hl)
      
      wl = line.strip()
      if i==0 and fls:
        wl = '*%s*'%wl
      for ext in ['py','txt']:
        if '.%s - '%ext in wl:
          s = wl.split('.%s - '%ext, 1)
          wl = '`' + s[0] + '.%s` - '%ext + s[1] + '\n'
      wiki.append(wl)
        

    self.html.write(self.doc.bigsection(title, '#ffffff', '#7799ee', '<br/>'.join(html)))
    
    self.wiki.write('== %s ==\n'%title)
    self.wiki.write('\n'.join(wiki))
    self.wiki.write('----\n\n')
  
  
  def addVariable(self, var, desc):
    """Adds a variable to the documentation. Given the nature of this you provide it as a pair of strings - one referencing the variable, the other some kind of description of its use etc.."""
    self.html_variables += '<strong>%s</strong><br/>'%var
    self.html_variables += '%s<br/><br/>\n'%desc
    
    self.wiki_variables += '*`%s`*\n'%var
    self.wiki_variables += '    %s\n\n'%desc


  def addFunction(self, func):
    """Adds a function to the documentation. You provide the actual function instance."""
    self.html_functions += self.doc.docroutine(func).replace('&nbsp;',' ')
    self.html_functions += '\n'
    
    name = func.__name__
    args, varargs, keywords, defaults = inspect.getargspec(func)
    doc = inspect.getdoc(func)
    
    if defaults==None: defaults = list()
    defaults = (len(args)-len(defaults)) * [None] + list(defaults)
    
    arg_str = ''
    if len(args)!=0:
      arg_str += reduce(lambda a, b: '%s, %s'%(a,b), map(lambda arg, d: arg if d==None else '%s = %s'%(arg,d), args, defaults))
      
    if varargs!=None:
      arg_str += ', *%s'%varargs if arg_str!='' else '*%s'%varargs
    if keywords!=None:
      arg_str += ', **%s'%keywords if arg_str!='' else '**%s'%keywords
    
    self.wiki_functions += '*`%s(%s)`*\n'%(name, arg_str)
    self.wiki_functions += '    %s\n\n'%doc
  
  
  def addClass(self, cls):
    """Adds a class to the documentation. You provide the actual class object."""
    self.html_classes += self.doc.docclass(cls).replace('&nbsp;',' ')
    self.html_classes += '\n'
    
    name = cls.__name__
    parents = filter(lambda a: a!=cls, inspect.getmro(cls))
    doc = inspect.getdoc(cls)
    
    par_str = ''
    if len(parents)!=0:
      par_str += reduce(lambda a, b: '%s, %s'%(a,b), map(lambda p: p.__name__, parents))
    
    self.wiki_classes += '== %s(%s) ==\n'%(name, par_str)
    self.wiki_classes += '    %s\n\n'%doc
    
    methods = inspect.getmembers(cls, lambda x: inspect.ismethod(x) or inspect.isbuiltin(x) or inspect.isroutine(x))
    def method_key(pair):
      if pair[0]=='__init__': return '___'
      else: return pair[0]
    methods.sort(key=method_key)
    
    
    for name, method in methods:
      if not name.startswith('_%s'%cls.__name__) and (not inspect.ismethod(method) and name[:2]!='__'):
        if inspect.ismethod(method):
          args, varargs, keywords, defaults = inspect.getargspec(method)
        else:
          args = ['?']
          varargs = None
          keywords = None
          defaults = None
        
        if defaults==None: defaults = list()
        defaults = (len(args)-len(defaults)) * [None] + list(defaults)
    
        arg_str = ''
        if len(args)!=0:
          arg_str += reduce(lambda a, b: '%s, %s'%(a,b), map(lambda arg, d: arg if d==None else '%s = %s'%(arg,d), args, defaults))
      
        if varargs!=None:
          arg_str += ', *%s'%varargs if arg_str!='' else '*%s'%varargs
        if keywords!=None:
          arg_str += ', **%s'%keywords if arg_str!='' else '**%s'%keywords
        
        def fetch_doc(cls, name):
          try:
            method = getattr(cls, name)
            if method.__doc__!=None: return inspect.getdoc(method)
          except: pass
          
          for parent in filter(lambda a: a!=cls, inspect.getmro(cls)):
            ret = fetch_doc(parent, name)
            if ret!=None: return ret
            
          return None

        doc = fetch_doc(cls, name)
        
        self.wiki_classes += '*`%s(%s)`*\n'%(name, arg_str)
        self.wiki_classes += '    %s\n\n'%doc
    
    
    variables = inspect.getmembers(cls, lambda x: inspect.ismemberdescriptor(x) or isinstance(x, int) or isinstance(x, str) or isinstance(x, float))
    
    for name, var in variables:
      if not name.startswith('__'):
        if hasattr(var, '__doc__'): d = var.__doc__
        else: d = str(var)
        self.wiki_classes += '*`%s`* = %s\n\n'%(name, d)
