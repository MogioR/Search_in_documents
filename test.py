from Modules.MemoryArray import MemoryArray
import struct

print(float(1000).__sizeof__())

test_memory = MemoryArray('test.bin', [100, 50], 4)

for i in range(0, 99):
    for j in range(0, 49):
        test_memory.set([i, j], struct.pack('>f', float(i+0.01*j)))

for i in range(0, 99):
    for j in range(0, 49):
        print(struct.unpack('>f', test_memory.get([i, j]))[0], end=' ')
    print()