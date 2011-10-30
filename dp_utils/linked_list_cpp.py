# Copyright (c) 2011, Tom SF Haines
# All rights reserved.

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#  * Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL <COPYRIGHT HOLDER> BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.



from scipy import weave
import unittest
from utils.start_cpp import start_cpp



# Defines code for a doubly linked list - simple but works as expected... (Includes its data via templated inheritance - a little strange, but neat and saves on memory thrashing.)
linked_list_code = start_cpp() + """
// Predefinitions...
template <typename ITEM, typename BODY> class Item;
template <typename ITEM, typename BODY> class List;

// Useful default...
struct Empty {};



// Item for the linked list data structure - simply inherits extra data stuff...
template <typename ITEM = Empty, typename BODY = Empty>
class Item : public ITEM
{
 public:
   Item(List<ITEM,BODY> * head):head(head),next(this),prev(this) {}
  ~Item() {}

  Item<ITEM,BODY> * Next() {return next;}
  Item<ITEM,BODY> * Prev() {return prev;}
  List<ITEM,BODY> * GetList() {return head;}

  bool Valid() {return static_cast< Item<ITEM,BODY>* >(head)!=this;}
  bool IsDummy() {return static_cast< Item<ITEM,BODY>* >(head)==this;}

  Item<ITEM,BODY> * PreNew() // Adds a new item before this one.
  {
   Item<ITEM,BODY> * ret = new Item<ITEM,BODY>(head);
   head->size += 1;
   ret->prev = this->prev;
   ret->next = this;
   ret->prev->next = ret;
   ret->next->prev = ret;
   return ret;
  }

  Item<ITEM,BODY> * PostNew() // Adds a new item after this one.
  {
   Item<ITEM,BODY> * ret = new Item<ITEM,BODY>(head);
   head->size += 1;
   ret->prev = this;
   ret->next = this->next;
   ret->prev->next = ret;
   ret->next->prev = ret;
   return ret;
  }
  
  void Suicide() // Removes this node from its list and makes it delete itself.
  {
   head->size -= 1;
   next->prev = prev;
   prev->next = next;
   delete this;
  }

 protected:
  List<ITEM,BODY> * head;
  Item<ITEM,BODY> * next;
  Item<ITEM,BODY> * prev;
};



// Simple totally inline doubly linked list structure, where
template <typename ITEM = Empty, typename BODY = Empty>
class List : protected Item<ITEM,BODY>
{
 public:
   List():Item<ITEM,BODY>(this),size(0) {}
  ~List()
  {
   while(this->size!=0)
   {
    this->next->Suicide();
   }
  }

  Item<ITEM,BODY> * Append() {return this->PreNew();}
  Item<ITEM,BODY> * Prepend() {return this->PostNew();}
  Item<ITEM,BODY> * First() {return this->next;}
  Item<ITEM,BODY> * Last() {return this->prev;}
  int Size() {return this->size;}
  BODY & Body() {return body;}

  Item<ITEM,BODY> * Index(int i)
  {
   Item<ITEM,BODY> * ret = this->next;
   while(i>0)
   {
    ret = ret->next;
    i -= 1;
   }
   return ret;
  }
   

 protected:
  friend class Item<ITEM,BODY>;
  int size;
  BODY body;
};

"""



class TestLinkedList(unittest.TestCase):
  """Test code for the linked list."""
  def test_compile(self):
    code = start_cpp(linked_list) + """
    """
    weave.inline(code, support_code=linked_list)

  def test_size(self):
    code = start_cpp(linked_list) + """
    int errors = 0;

    List<> wibble;
    if (wibble.Size()!=0) errors += 1;
    Item<> * it = wibble.Append();
    if (wibble.Size()!=1) errors += 1;
    it->Suicide();
    if (wibble.Size()!=0) errors += 1;
    
    return_val = errors;
    """
    
    errors = weave.inline(code, support_code=linked_list)
    self.assertEqual(errors,0)

  def test_loop(self):
    extra = """
    struct Number
    {
     int num;
    };
    """
    
    code = start_cpp(linked_list_code+extra) + """
    int errors = 0;

    List<Number> wibble;
    for (int i=0;i<10;i++)
    {
     Item<Number> * it = wibble.Append();
     it->num = i;
    }
    if (wibble.Size()!=10) errors += 1;

    int i = 0;
    for (Item<Number> * targ = wibble.First(); targ->Valid(); targ = targ->Next())
    {
     if (i!=targ->num) errors += 1;
     i += 1;
    }

    return_val = errors;
    """
    
    errors = weave.inline(code, support_code=linked_list_code+extra)
    self.assertEqual(errors,0)



# Code for a linked list with garbage collection - each entry has a reference count, and it also allows access of the reference counts and the total number of reference counts for all entrys. This structure is very useful for modelling a Dirichlet process as a direct consequence, as it has all its properties...
linked_list_gc_code = linked_list_code + start_cpp() + """
// Predefinitions...
template <typename ITEM, typename BODY> class ItemRef;
template <typename ITEM, typename BODY> class ListRef;



// Item for the linked list data structure - simply inherits extra data stuff...
template <typename ITEM = Empty, typename BODY = Empty>
class ItemRef : public ITEM
{
 public:
   ItemRef(ListRef<ITEM,BODY> * head):head(head),next(this),prev(this),refCount(0) {}
  ~ItemRef() {}

  ItemRef<ITEM,BODY> * Next() {return next;}
  ItemRef<ITEM,BODY> * Prev() {return prev;}
  ListRef<ITEM,BODY> * GetList() {return head;}

  bool Valid() {return static_cast< ItemRef<ITEM,BODY>* >(head)!=this;}
  bool IsDummy() {return static_cast< ItemRef<ITEM,BODY>* >(head)==this;}

  ItemRef<ITEM,BODY> * PreNew() // Adds a new item before this one.
  {
   ItemRef<ITEM,BODY> * ret = new ItemRef<ITEM,BODY>(head);
   head->size += 1;
   ret->prev = this->prev;
   ret->next = this;
   ret->prev->next = ret;
   ret->next->prev = ret;
   return ret;
  }

  ItemRef<ITEM,BODY> * PostNew() // Adds a new item after this one.
  {
   ItemRef<ITEM,BODY> * ret = new ItemRef<ITEM,BODY>(head);
   head->size += 1;
   ret->prev = this;
   ret->next = this->next;
   ret->prev->next = ret;
   ret->next->prev = ret;
   return ret;
  }

  void Suicide() // Removes this node from its list and makes it delete itself.
  {
   head->size -= 1;
   head->refTotal -= refCount;
   next->prev = prev;
   prev->next = next;

   delete this;
  }

  void IncRef(int amount = 1)
  {
   this->refCount += amount;
   head->refTotal += amount;
  }

  void DecRef(int amount = 1) // If the ref count reaches zero the object will delete itself.
  {
   this->refCount -= amount;
   head->refTotal -= amount;
   if (refCount<=0) this->Suicide();
  }

  int RefCount() {return refCount;}

 protected:
  ListRef<ITEM,BODY> * head;
  ItemRef<ITEM,BODY> * next;
  ItemRef<ITEM,BODY> * prev;
  int refCount;
};



// Simple totally inline doubly linked list structure...
template <typename ITEM = Empty, typename BODY = Empty>
class ListRef : protected ItemRef<ITEM,BODY>
{
 public:
   ListRef():ItemRef<ITEM,BODY>(this),size(0),refTotal(0) {}
  ~ListRef()
  {
   while(this->size!=0)
   {
    this->next->Suicide();
   }
  }

  ItemRef<ITEM,BODY> * Append() {return this->PreNew();}
  ItemRef<ITEM,BODY> * Prepend() {return this->PostNew();}
  ItemRef<ITEM,BODY> * First() {return this->next;}
  ItemRef<ITEM,BODY> * Last() {return this->prev;}
  int Size() {return this->size;}
  int RefTotal() {return this->refTotal;}
  BODY & Body() {return body;}
  
  ItemRef<ITEM,BODY> * Index(int i)
  {
   ItemRef<ITEM,BODY> * ret = this->next;
   while(i>0)
   {
    ret = ret->Next();
    i -= 1;
   }
   return ret;
  }
  
 protected:
  friend class ItemRef<ITEM,BODY>;
  int size;
  int refTotal;
  BODY body;
};
"""



class TestLinkedListGC(unittest.TestCase):
  """Test code for the linked list with garbage collection."""
  def test_compile(self):
    code = start_cpp(linked_list_gc) + """
    """
    weave.inline(code, support_code=linked_list_gc)

  def test_size_gc(self):
    code = start_cpp(linked_list_gc_code) + """
    int errors = 0;

    ListRef<> wibble;
    if (wibble.Size()!=0) errors += 1;
    ItemRef<> * it = wibble.Append();
    if (wibble.Size()!=1) errors += 1;
    if (wibble.RefTotal()!=0) errors += 1;

    it->IncRef();
    it->IncRef();
    if (it->RefCount()!=2) errors += 1;
    if (wibble.RefTotal()!=2) errors += 1;

    it->DecRef();
    it->DecRef();
    if (wibble.RefTotal()!=0) errors += 1;
    if (wibble.Size()!=0) errors += 1;

    return_val = errors;
    """

    errors = weave.inline(code, support_code=linked_list_gc_code)
    self.assertEqual(errors,0)



# If this file is run do the unit tests...
if __name__ == '__main__':
  unittest.main()
