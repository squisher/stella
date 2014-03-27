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
        return self.backend[-1]
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
                self.next = self.stack.pop().next
                return self.__next__()
            raise StopIteration()

        if isinstance(self.next, Block):
            # TODO this doesn't allow empty blocks
            current = self.next._block_start.next
            self.stack.push(self.next)
        else:
            current = self.next


        self.next = current.next
        return current

def linkedlist(klass):
    klass.next = None
    klass.prev = None

    def __iter__(self):
        return LinkedListIter(self)
    klass.__iter__ = __iter__

    def printAll(self):
        """Debugging: print all IRs in this list"""

        # find the first bytecode
        bc_start = self
        while bc_start.prev != None:
            bc_start = bc_start.prev

        for bc in bc_start:
            logging.debug(str(bc))
    klass.printAll = printAll
    
    def insert_after(self, bc):
        self.next = bc
        bc.prev = self
    klass.insert_after = insert_after

    def insert_before(self, bc):
        bc.prev = self.prev
        bc.next = self

        bc.prev.next = bc
        self.prev = bc
    klass.insert_before = insert_before

    def remove(self):
        if self.prev:
            self.prev.next = self.next
        if self.next:
            self.next.prev = self.prev
    klass.remove = remove

    return klass

@linkedlist
class Block(object):
    """A block is a nested list of bytecodes."""
    _block_start = None
    def __init__(self, bc):
        self._block_start = bc

    def blockContent(self):
        return self._block_start

@linkedlist
class BlockStart(object):
    """Dummy start object, ignore when iterating.
    
    This is only used to make the insertion code easier."""
    pass

class BlockEnd(object):
    """Marks the end of a block of nested bytecodes.
    
    Enables checks via multiple inheritance."""
    pass
