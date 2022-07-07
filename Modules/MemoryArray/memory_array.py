import os
import struct


class MemoryArray:
    def __init__(self, name: str, directions: list, size_of_element: int):
        """
        :param name: root of file in memory
        :param directions: sizes of array in memory
        :param size_of_element: size of one element
        """
        self.name = name
        self.directions = directions
        self.size_of_element = size_of_element
        self.file_pointer = None

    def load(self, root=''):
        self.file_pointer = open(root + self.name, 'rb+')

    def create(self, root=''):
        self.file_pointer = open(root + self.name, 'wb+')

        # File generate
        size = self.size_of_element
        for direction in self.directions:
            size *= direction

        allocated = 0
        size_to_allocation = 1000000  # One megabyte
        while size > allocated:
            if size < size_to_allocation:
                size_to_allocation = size
            self.file_pointer.write(("\0" * size_to_allocation).encode())
            allocated += size_to_allocation
            # self.file_pointer.seek(allocated)

    # Set element in memory
    def set(self, pos, element):
        # if element.__sizeof__() > self.size_of_element:
        #     raise ValueError('Wrong size of element, max size is {0}, element size is {1}' \
        #                      .format(self.size_of_element, element.__sizeof__()))
        flat_pos = self.get_flat_pos(pos)
        self.file_pointer.seek(flat_pos)
        self.file_pointer.write(element)

    # Get element from memory
    def get(self, pos):
        flat_pos = self.get_flat_pos(pos)
        self.file_pointer.seek(flat_pos)
        return self.file_pointer.read(self.size_of_element)

    # Transform vector pos to float cord in memory
    def get_flat_pos(self, pos: list) -> int:
        flat_pos = 0
        direction_size = self.size_of_element

        if len(pos) != len(self.directions):
            raise ValueError('Directions count is {0}'.format(len(self.directions)))

        for num, direction_pos in enumerate(pos):
            if direction_pos >= self.directions[num]:
                raise IndexError('Direction {0} size is {1}, tried index is {2}' \
                                 .format(num, self.directions[num], direction_pos))

            flat_pos += direction_pos * direction_size
            direction_size *= self.directions[num]

        return flat_pos

    def __del__(self):
        self.file_pointer.close()
        # os.remove(self.name)
