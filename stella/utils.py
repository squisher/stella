import logging

class Stack(object):
    backend = None
    def __init__(self, name="Stack"):
        self.backend = []
        self.name = name
    def __str__(self):
        return "["+self.name+"("+str(len(self.backend))+")]"
    def __repr__(self):
        return "["+self.name+"="+", ".join([str(x) for x in self.backend])+"]"
    def push(self, item):
        logging.debug("["+self.name+"] Pushing " + str(item))
        self.backend.append(item)
    def pop(self):
        item = self.backend.pop()
        logging.debug("["+self.name+"] Popping " + str(item))
        return item
    def peek(self):
        if len(self.backend) > 0:
            return self.backend[-1]
        else:
            return None
    def empty(self):
        return len(self.backend) == 0
    def clone(self):
        s = Stack(self.name)
        s.backend = [x for x in self.backend]
        return s

class LinkedListIter(object):
    def __init__(self, start):
        self.next = start
        self.stack = Stack('iter')

    def __iter__(self):
        return self

    def __next__(self):
        if self.next == None:
            if not self.stack.empty():
                #import pdb; pdb.set_trace()
                self.next = self.stack.pop()
                return self.__next__()
            raise StopIteration()

        if isinstance(self.next, Block):
            self.stack.push(self.next.next)
            self.next = self.next._block_start
            return self.__next__()

        current = self.next
        self.next = self.next.next
        return current

def linkedlist(klass):
    klass.next = None
    klass.prev = None
    klass._block_parent = None

    def __iter__(self):
        return LinkedListIter(self)
    klass.__iter__ = __iter__

    def printAll(self):
        """Debugging: print all IRs in this list"""

        # find the first bytecode
        bc_start = self
        while True:
            while bc_start.prev != None:
                bc_start = bc_start.prev
            if bc_start._block_parent == None:
                break
            else:
                bc_start = bc_start._block_parent

        for bc in bc_start:
            #logging.debug(str(bc))
            logging.debug(bc.locStr())
    klass.printAll = printAll
    
    def insert_after(self, bc):
        """Insert bc after self.

        Note: block start and end are not adjusted here! They're only checked at remove()"""
        bc.next = self.next
        if bc.next:
            bc.next.prev = bc
        self.next = bc
        bc.prev = self
    klass.insert_after = insert_after

    def insert_before(self, bc):
        """Insert bc before self.

        Note: block start and end are not adjusted here! They're only checked at remove()"""
        bc.prev = self.prev
        bc.next = self

        bc.prev.next = bc
        self.prev = bc
    klass.insert_before = insert_before

    def remove(self):
        if self.next:
            self.next.prev = self.prev
            if self.blockStart():
                # Move the block start attribute over to the next
                self.next.blockStart(self.blockStart())
        if self.prev:
            self.prev.next = self.next
            if self.blockEnd():
                # Move the block end attribute over to the prev
                self.prev.blockEnd(self.blockEnd())
    klass.remove = remove

    def blockStart(self, new_parent = None):
        """Get the block parent, or set a new block parent."""
        if new_parent == None:
            return self._block_parent

        # Update the block's start
        new_parent._block_start = self
        # Remember the block
        self._block_parent = new_parent
    klass.blockStart = blockStart

    def blockEnd(self, new_parent = None):
        """Get the block parent, or set a new block parent."""
        if new_parent == None:
            return self._block_parent

        # Update the block's end
        new_parent._block_end = self
        # Remember the block
        self._block_parent = new_parent
    klass.blockEnd = blockEnd

    def linearNext(self):
        """Move to the previous bytecode, transparently handling blocks"""
        # TODO should this be its own iterator?
        if self.next == None:
            if self._block_parent:
                return self._block_parent.next
            else:
                return None
        if isinstance(self.next, Block):
            return self.next.blockContent()
        return self.next
    klass.linearNext = linearNext

    def linearPrev(self):
        """Move to the previous bytecode, transparently handling blocks"""
        # TODO should this be its own iterator?
        if self.prev == None:
            if self._block_parent:
                return self._block_parent.prev
            else:
                return None
        if isinstance(self.prev, Block):
            return self.prev._block_end
        return self.prev
    klass.linearPrev = linearPrev

    return klass

@linkedlist
class Block(object):
    """A block is a nested list of bytecodes."""
    _block_start = None
    _block_end = None
    def __init__(self, bc):
        self._block_start = bc
        bc._block_parent = self

    def blockContent(self):
        return self._block_start

@linkedlist
class BlockStart(object):
    """Marks the start of a block of nested bytecodes.
    
    Enables checks via multiple inheritance."""
    pass

class BlockEnd(object):
    """Marks the end of a block of nested bytecodes.
    
    Enables checks via multiple inheritance."""
    pass