import sys
import numpy as np
from numba import jit
python3 = sys.version_info.major >= 3

CODEBITS=24
TOP=(1<<CODEBITS)
MASK32 = 0xFFFFFFFF
BOTTOM=(TOP>>8)

@jit(nopython=True) 
def update_cum_freq2(freq, symbol):
    cumfreqsymbol = 0
    for i in range(symbol):
        cumfreqsymbol = cumfreqsymbol + freq[i]
    return cumfreqsymbol


# Provides the state and behaviors that arithmetic coding encoders and decoders share.
class ArithmeticCoderBase(object):

    # Constructs an arithmetic coder, which initializes the code range.
    def __init__(self, numbits):
        self.low = np.int64(0)
        self.range = np.int64(TOP)

# Encodes symbols and writes to an arithmetic-coded bit stream.
class ArithmeticEncoder(ArithmeticCoderBase):
    def __init__(self, numbits, bitout, buffer_size=4096):
        super(ArithmeticEncoder, self).__init__(numbits)
        self.output = bitout
        self.write_buffer = bytearray(buffer_size)
        self.buffer_pos = 0
        self.buffer_size = buffer_size

    def _flush_write_buffer(self):
        if self.buffer_pos > 0:
            self.output.write(self.write_buffer[:self.buffer_pos])
            self.buffer_pos = 0

    def _write_byte(self, byte):
        self.write_buffer[self.buffer_pos] = byte
        self.buffer_pos += 1
        if self.buffer_pos >= self.buffer_size:
            self._flush_write_buffer()

    def write(self, freqs, symbol,alphabet_size):
        total = freqs[alphabet_size]
        cumfreqsymbol= update_cum_freq2(freqs, symbol)
        #print('symbol=%i, cumfreqsymbol=%i, total=%i' % (symbol, cumfreqsymbol, total))
        self.range = self.range // total
        self.low = self.low + cumfreqsymbol * self.range
        self.range = self.range * freqs[symbol]

        while True:
            if (self.low ^ (self.low + self.range)) < TOP:
                self._write_byte((self.low >> 24) & 0xFF)
                self.range = (self.range << 8) & MASK32
                self.low = (self.low << 8) & MASK32
            elif self.range < BOTTOM:
                self.range = ((-self.low) & (BOTTOM - 1)) & MASK32
                self._write_byte((self.low >> 24) & 0xFF)
                self.range = (self.range << 8) & MASK32
                self.low = (self.low << 8) & MASK32
            else:
                break  



    def finish(self):
        for _ in range(4):
            self._write_byte((self.low >> 24)& 0xFF)  
            self.low = (self.low << 8) & MASK32
        self._flush_write_buffer()  

# Reads from an arithmetic-coded bit stream and decodes symbols.
class ArithmeticDecoder(ArithmeticCoderBase):

    def __init__(self, numbits, bitin):
        super(ArithmeticDecoder, self).__init__(numbits)
        self.input = bitin
        self.code = 0
        for _ in range(4):
            streambyte=self.input.read(1)
            self.code = ((self.code<<8) | streambyte[0]) & MASK32

    def updatedec(self, freqs,cumfreqs, symbol,alphabet_size):
        total = cumfreqs[alphabet_size] 
        self.range = self.range // total
        self.low = self.low +  cumfreqs[symbol] * self.range
        self.range = self.range * freqs[symbol]
        while True:
            if (self.low ^ (self.low + self.range)) < TOP:
                streambyte = self.input.read(1)
                self.code = ((self.code << 8) | streambyte[0]) & MASK32
                self.range = (self.range << 8) & MASK32
                self.low = (self.low << 8) & MASK32
            elif self.range < BOTTOM:
                self.range = ((-self.low) & (BOTTOM - 1)) & MASK32
                streambyte = self.input.read(1)
                self.code = ((self.code << 8) | streambyte[0]) & MASK32
                self.range = (self.range << 8) & MASK32
                self.low = (self.low << 8) & MASK32
            else:
                break  
    def read(self, freqs, cumfreqs,alphabet_size):
        total = cumfreqs[alphabet_size]
        range32 = (self.range // total) & MASK32
        offset = (self.code - self.low) & MASK32
        value = offset // range32 if range32 > 0 else 0
        
        symbol = 0
        while symbol < alphabet_size and cumfreqs[symbol + 1] <= value:
            symbol += 1
        
        #low_idx = 0
        #high_idx = 256  # ALPHABET_SIZE
        #while low_idx < high_idx:
        #    mid = (low_idx + high_idx) // 2
        #    if cumfreqs[mid] <= value:
        #        low_idx = mid + 1
        #    else:
        #        high_idx = mid
        #symbol = low_idx - 1

        
        self.updatedec(freqs, cumfreqs, symbol,alphabet_size)
        
        return symbol
