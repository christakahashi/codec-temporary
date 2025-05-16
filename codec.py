import os
import struct
import itertools
import logging
from typing import Optional,Union, Callable, Tuple  #requires python 3.5 or later


import numpy as np
from numpy.typing import ArrayLike

import galois  
import reedsolo 

#### ROADMAP ####
# - integrate the baseN to DNA function here.  (generalize from b32fn)
# - Strand prefix sub-encoder.
# - Sample objective functions (at least minimize homopolymers and avoid annealing.)


lipsum = b"Lorem ipsum dolor sit amet, consectetuer adipiscing elit. Aenean commodo ligula eget dolor. Aenean massa. Cum sociis natoque penatibus et magnis dis parturient montes, nascetur ridiculus mus. Donec quam felis, ultricies nec, pellentesque eu, pretium quis, sem. Nulla consequat massa quis enim. Donec pede justo, fringilla vel, aliquet nec, vulputate eget, arcu. In enim justo, rhoncus ut, imperdiet a, venenatis vitae, justo. Nullam dictum felis eu pede mollis pretium. Integer tincidunt. Cras dapibus. Vivamus elementum semper nisi. Aenean vulputate eleifend tellus. Aenean leo ligula, porttitor eu, consequat vitae, eleifend ac, enim. Aliquam lorem ante, dapibus in, viverra quis, feugiat a, tellus. Phasellus viverra nulla ut metus varius laoreet. Quisque rutrum. Aenean imperdiet. Etiam ultricies nisi vel augue. Curabitur ullamcorper ultricies nisi. Nam eget dui. Etiam rhoncus. Maecenas tempus, tellus eget condimentum rhoncus, sem quam semper libero, sit amet adipiscing sem neque sed ipsum." #cspell:disable-line
  

def _int_to_baseN(n:int,base:int,length:int=-1)->list[int]:
    """Converts an integer to a base N number in little endian order.

    Args:
        n (int): The integer to be converted.
        base (int): The base of the number system.
        length (int, optional): If specified, the number will be zero padded to this length.

    Returns:
        list[int]: The list of digits representing the base N number.

    """ 

    digits = []
    while n>0:
        n,r = divmod(n,base)
        digits.append(r)
        
    #do zero padding
    while len(digits)<length: 
        digits.append(0)
    return digits

def _baseN_to_int(digits: list[int], base: int) -> int:
    """
    Converts a base N number to an integer using little endian order.

    Args:
        digits (list[int]): The list of digits representing the base N number.
        base (int): The base of the number.

    Returns:
        int: The converted integer value.

    """

    n = 0
    for d in reversed(digits):
        n = n*base + d
    return n

def computeNStrands(nbytes:int, 
                 inner_alphabet_size:int=32, 
                 inner_d:int=5, 
                 inner_n:Optional[int]=None, 
                 n_redundant_strands:int=5)->int:
    
    #TODO: do some algebra to solve for n_strands.
    min_m=1
    max_m=2**16-1
    while min_m<max_m:
        mid = (min_m+max_m)//2
        codec = BaseNBlockCodec(inner_alphabet_size=inner_alphabet_size,
                                inner_d=inner_d,
                                inner_n=inner_n,
                                n_redundant_strands=n_redundant_strands,
                                n_strands=mid)
        if codec.block_capacity_bytes < nbytes:
            min_m = mid+1
        else:
            max_m = mid
    return min_m
    #index_bits = np.ceil(np.log2(n_strands)).astype(int)
    #k = inner_n - (inner_d - 1)
    #data_chunk_size = int(k*np.log2(inner_alphabet_size)-index_bits) // 8
    #block_capacity_bytes = data_chunk_size*(n_strands - n_redundant_strands)




class BaseNBlockCodec:
    def __init__(self, 
                 inner_alphabet_size:int=32, 
                 inner_d:int=5, 
                 inner_n:Optional[int]=None, 
                 n_redundant_strands:int=5, 
                 n_strands:int=100,
                 max_strand_index:Optional[int]=None): 
        """  
        Initialize the Codec object.

        Args:
            inner_alphabet_size (int): Size of the inner alphabet (must be a prime power).
            inner_d (int): Inner codeword distance.
            inner_n (Optional[int]): Number of symbols in inner block (n < inner_alphabet_size). Default: inner_alphabet_size-1.
            n_redundant_strands (int): Number of redundant strands.
            n_strands (int): Number of "strands" stored by this codec including redundant strands.
            max_strand_index (Optional[int]): Maximum guaranteed index. Default: n_strands.
        """
        #TODO: handle more than 255 total strands.
        #this was formally a parameter but probably better to keep it fixed or automatically calculated.
        if n_strands<=255:
            self.outer_alphabet_size_bytes = 1
        else:
            self.outer_alphabet_size_bytes = 2 
        if n_strands>(2**(8*self.outer_alphabet_size_bytes)-1):
            raise ValueError("Too many strands for outer code.")
        self.n_strands = n_strands

        self.n_message_strands = n_strands - n_redundant_strands

        if inner_n is None:
            inner_n = inner_alphabet_size - 1
        else:
            #TODO: if inner_n is requested to be too large, double up on inner blocks for a single "strand"
            #      make sure extra sequence numbers are not added to inner blocks.  and or maybe add a 
            #      parameter for the number of inner blocks per strand.
            assert inner_n<inner_alphabet_size
        
        self.inner_coder = galois.ReedSolomon(n=inner_alphabet_size-1,d=inner_d,field=galois.GF(inner_alphabet_size))
        #  in units of symbols (there are q=inner_alphabet_size possible symbols)
        #  ##################strand (n symbols)#######################
        #  # message                         # index #   redundancy  #
        #  |<----------------- k ------------------->|<---- d-1 ---->|
        #  |<-----------------------  n ---------------------------->|
        #
        #  sizes in bytes:
        #  |<---- self.data_chunk_size ----->|<- . ->|
        #                                        ^ = self.index_bits <-- in bits not bytes.
        #  note: actual k remains alphabet_size-1, but the code is shorted (0 padded) to inner_n symbols. 
        k = inner_n - (inner_d - 1) 
        q = self.inner_coder.field.order

        #number of bytes needed to store the index
        self.index_bits = np.ceil(np.log2(self.n_strands)).astype(int)
        logging.debug("bits per strand used for indexing: {}".format(self.index_bits))

        #message length in bytes (without index)
        self.data_chunk_size = int(k*np.log2(q)-self.index_bits) // 8
        logging.debug( "strand capacity in bytes: {}".format(self.data_chunk_size) )

        _wasted_bits = k*np.log2(q) - (self.data_chunk_size*8+self.index_bits )
        logging.debug( "{} bits wasted per strand".format( _wasted_bits ) )
        logging.debug( "Maximum strand index (using wasted bits): {}".format(2**(int(self.index_bits+_wasted_bits))))
        _wasted_symbols = _wasted_bits/np.log2(q)
        if  _wasted_symbols>=1:
            logging.warning( "{} symbols wasted per strand.  Consider decreasing inner_n.".format( _wasted_symbols) )
        else:
            logging.debug("{} symbols wasted per strand".format( _wasted_symbols))
 
         
        self.block_capacity_bytes = self.data_chunk_size*(self.n_strands - n_redundant_strands) 
        logging.debug( "codec message size: {} bytes".format(self.block_capacity_bytes) )

        #init the outer coder 
        self.outer_coder = reedsolo.RSCodec(n_redundant_strands, c_exp=8*self.outer_alphabet_size_bytes)

    def encode(self,data:bytes,index_start:int=0)->list[list[int]]:
        """ Encodes the given data using the outer and inner codes.

        Args:
            data (bytes): The input data to be encoded.
            index_start (int, optional): The starting index for the encoded chunks. Defaults to 0.

        Returns:
            list[list[int]]: The encoded chunks of data.
        """

        #TODO: add option to put the index in multiples of inner symbols instead of bytes.
        #      this would give options to improve data packing depending on parameters.
        #      even better this could be autocomputed.
        #TODO: consider putting outer code redundancy striped vertically bytewise so that
        #      lost strands with odd numbers of bytes don't erase bytes from neighboring strands.


        #encode process outline:
        #  0. serial data 
        #  |--------------------------------... -----------|
        #  1. strand length payload chunks (self.data_chunk_size in bytes)
        #  |<----------------- self.data_chunk_size ------------->|
        #  |<----------------- self.data_chunk_size ------------->|
        #  ...
        #  |<----------------- self.data_chunk_size ------------->|
        #  2. outer code  (r= outer code redundancy applied to columns)
        #  |<----------------- self.data_chunk_size ------------->|
        #  |<----------------- self.data_chunk_size ------------->|
        #  ...
        #  |<----------------- self.data_chunk_size ------------->|
        #  |<rrrrrrrrrrrrrrrrr self.data_chunk_size rrrrrrrrrrrrr>|
        #  ...
        #  |<rrrrrrrrrrrrrrrrr self.data_chunk_size rrrrrrrrrrrrr>|
        #  3. inner code  (R= inner code redundancy applied to rows, * = inner code symbol)
        #  |<***************** self.data_chunk_size *************RRRRRRRR>|
        #  |<***************** self.data_chunk_size *************RRRRRRRR>|
        #  ...
        #  |<***************** self.data_chunk_size *************RRRRRRRR>|
        #  |<rrrrrrrrrrrrrrrrr self.data_chunk_size rrrrrrrrrrrrrRRRRRRRR>|
        #  ...
        #  |<rrrrrrrrrrrrrrrrr self.data_chunk_size rrrrrrrrrrrrrRRRRRRRR>|

        #1. Reshape data 
        logging.debug("applying outer code to {} bytes".format(len(data)))
        # reshape data so columns can be coded independently.
        if self.outer_coder.c_exp==8:
            data_np = np.frombuffer(data,dtype=np.uint8)
            assert len(data_np)%self.data_chunk_size == 0
            data_np = data_np.reshape(-1,self.data_chunk_size)
        elif self.outer_coder.c_exp==16:
            data_np = np.frombuffer(data,dtype=np.uint16)
            if self.data_chunk_size%2!=0:
                raise ValueError("data_chunk_size must be even for 16 bit outer code.")
            if len(data_np)%(self.data_chunk_size//2) != 0:
                raise ValueError(f"data length(={len(data_np)}) must be multiple of data_chunk_size//2(={self.data_chunk_size//2})."+
                                  " Consider padding data.") 
            data_np = data_np.reshape(-1,self.data_chunk_size//2)
        else:
            raise ValueError("Unsupported outer code byte size.")

        #2. apply the outer code (encode columns of data)
        data_enc_cols = np.array([self.outer_coder.encode(dc) for dc in data_np.transpose()],dtype=data_np.dtype)
        data_enc_chunks = data_enc_cols.transpose()
        
        logging.debug("outer coded strands: {}".format(data_enc_chunks.shape[0]))

        #3. apply the inner code
        chunks = []
        index = index_start
        _temp = 0
        for chunk in data_enc_chunks:
            #add index to "strand".
            d = chunk.tobytes()
            #use little endian to match x86/x64 byte order and avoid needing to pad byte string.
            chunk = int.from_bytes(d,'little') 
            #add index to end (MS bits)
            chunk += index<<(self.data_chunk_size*8)
            index += 1
            #but.. we still need to pad out the symbol stream.
            chunk = _int_to_baseN(chunk,self.inner_coder.field.order,length=self.inner_coder.k)
            chunks.append(chunk)

        #apply inner code
        ic_chunks= [self.inner_coder.encode(c).tolist() for c in chunks] 

        return ic_chunks

    def decode(self, data: list[list[int]], index_start: int = 0) -> Tuple[bytes,ArrayLike,ArrayLike,ArrayLike]:
        """
        Decodes the given data.

        Args:
            data (list[list[int]]): The encoded data to be decoded.
            index_start (int, optional): The starting index for decoding. Defaults to 0.

        Returns:
            bytes: The decoded data.
            outer erasures: list of erasures in outer code.
            outer errors: list of errors in outer code.
            inner errors: list of number of errors in inner code by strand index. -1 -> strand not present.

        Raises:
            None
        """
        # inner decode
        ## TODO: handle erasures in inner code.
        chunked_data = [self.inner_coder.decode(d,errors=True) for d in data]
        

        # handle outer decode
        ordered_chunks = [None] * self.n_strands
        chunk_errors = [-1] * self.n_strands
        for chunk in chunked_data:
            chunk_int = _baseN_to_int(chunk[0].tolist(), self.inner_coder.field.order)
            chunk_index = chunk_int >> (self.data_chunk_size * 8) - index_start
            _mask = (2 ** (self.data_chunk_size * 8)) - 1
            chunk_int = chunk_int & _mask
            chunk_bytes = chunk_int.to_bytes(self.data_chunk_size, 'little')
            #print(chunk_index)
            try:
              ordered_chunks[chunk_index] = chunk_bytes
              chunk_errors[chunk_index] = chunk[1]
            except IndexError:
              logging.info("strand index out of bounds:" + str(chunk_index))

        # identify erasures
        erasures = []
        for i, c in enumerate(ordered_chunks):
            if c is None:
                erasures.append(i)  # remember where erasure is
                ordered_chunks[i] = b'\x00' * self.data_chunk_size  # fill in erasures with zeros

        if self.outer_coder.c_exp==8:
            outer_dtype = np.uint8
        elif self.outer_coder.c_exp==16:
            outer_dtype = np.uint16
        else:
            raise ValueError("Unsupported outer code byte size.")
        data_np_encoded = np.array([np.frombuffer(c, dtype=outer_dtype) for c in ordered_chunks]).transpose()
        data_decoded_results = [self.outer_coder.decode(dc, erase_pos=erasures) for dc in data_np_encoded]
        data_decoded_chunks = np.array([d[0] for d in data_decoded_results],dtype=outer_dtype).transpose()
        data_decoded = data_decoded_chunks.flatten().tobytes()

        # report erasures (lost strands) in outer code
        logging.info("erasures (lost) chunks: {}".format(erasures))
        data_errata_col_pos = [e[2] for e in data_decoded_results]

        errors = []
        for i, pos in enumerate(data_errata_col_pos):
            for strand in pos:
                if (not(strand in erasures)):
                    errors.append((strand, i))
        errors.sort(key=lambda x: x[0])

        # unlikely to find an error since it would have to pass the inner code.
        for e in errors:
            if e[0] in erasures:
                pass
            else:
                logging.warning("Outer code errors found on strand {}, byte {}.".format(*e))

        return data_decoded, erasures, errors, chunk_errors


#convert base N into DNA.
_default_b32_alphabet = ['GCT', 'ACT', 'AGT', 'CTC', 'TGC', 'GAG', 'GCA', 'ATG', 'AGA', 'AGC', 'CAC', 'CAT', 'CAG', 'CTG', 'TCT', 'GAC', 'GTC', 'GTG', 'ACA', 'ACG', 'ATC', 'CTA', 'CGA', 'CGT', 'TAC', 'TAG', 'TCA', 'TCG', 'TGA', 'TGT', 'GAT', 'GTA']
_default_b32_alphabet_alt = ['ATA', 'AAC', 'CCT', 'TAT', 'CCG', 'TGG', 'CGC', 'CGG', 'GTT', 'GGT', 'GCG', 'TCC', 'CCA', 'GGA', 'ACC', 'AGG', 'TAA', 'GCC', 'AAA', 'CCC', 'GGG', 'GGC', 'TTA', 'TTT', 'GAA', 'AAG', 'CTT', 'TTG', 'TTC', 'AAT', 'CAA', 'ATT']

def find_avoid(seq:str, avoids:list)->tuple[int,int]:
    """find index and length of avoided sequence """
    for a in avoids:
        loc = seq.find(a)
        if loc>0:
            return loc,len(a)
    return -1,0

def longest_match(a:str,b:str):
    def _longest_match(a:str,b:str):
        max_match = 0
        for i in range(len(a)): 
            match = 0
            for offset in range(min(len(a)-i,len(b))):
                if a[i+offset] == b[offset]:
                    match +=1
                else:
                    match = 0
                if match > max_match:
                    max_match = match
        return max_match
    A = a.upper()
    B = b.upper()
    return max(_longest_match(A,B),_longest_match(B,A))

#Returns the reverse complement of a DNA sequence.
def reverse_complement(dna_sequence: str) -> str:
    complement = {'A': 'T', 'T': 'A', 'C': 'G', 'G': 'C', 'a': 't', 't': 'a', 'c': 'g', 'g': 'c'}
    reversed_sequence = dna_sequence[::-1]
    reverse_complement_sequence = ''.join(complement[nucleotide] for nucleotide in reversed_sequence)
    return reverse_complement_sequence

def longest_binder(a:str,b:str):
  A = reverse_complement(a) 
  return longest_match(A,b)



def b32_to_DNA_optimize(file_data:ArrayLike,words:list[str], alternate_words:list[str], mask:Union[ArrayLike,None]=None, penalty_fn:Union[Callable[[str],int],None]=None)->list[str]:
  """ 
    TODO: rename to bN_to_DNA_optimize, and add test.
    file_data: 2d list of integral types (symbols)
    mask: list of ints. 0 for word, 1 for alternate word, -1 for don't care.  None for no mask.
    penalty_fn: function that takes a DNA sequence and returns a score.  penalty_fn(DNA_seq:str)->int
  """
  return [b32_to_DNA_optimize_single(x,words,alternate_words,penalty_fn=penalty_fn) for x in file_data]


def b32_to_DNA_optimize_single(strand_data:ArrayLike, words:list[str], alternate_words:list[str], mask:Union[ArrayLike,None]=None ,penalty_fn=None)->Tuple[bytes,int]:
  """
    TODO: rename to bN_to_DNA_optimize_single
    TODO: add mask support.
    strand_data: 1d list of integral types
    mask: list of ints. 0 for word, 1 for alternate word, -1 for don't care.  None for no mask.
  """
  strand_data = np.array(strand_data)
  
  if mask is not None:
    mask = np.array(mask)
    _m = np.full(len(strand_data),-1,dtype=np.int8)
    _m[:len(mask)] = np.array(mask.copy()) #Probably don't need this copy.
    mask = _m
    mutable_ind = np.where(mask==-1)
    mask[mutable_ind] = 0
    #raise NotImplementedError("mask not implemented yet")
  else:
    mask = np.zeros(len(strand_data),dtype=np.uint8)
    mutable_ind = np.arange(len(strand_data))
    
  words = np.array(list(zip(words,alternate_words)),dtype="S")
  picks = np.zeros(len(strand_data),dtype=np.uint8)
  picks[0:len(mask)] = mask
  dna_seq = b"".join(words[strand_data,picks])
  if penalty_fn is None:
     return dna_seq,0
  score = penalty_fn(dna_seq)

  #greedy search, stop when you can't find a better solution one step away.
  #this may end on a local minimum.
  icount = 0
  while True:
    iteration_best_picks = picks.copy()
    iteration_best_score = score
    iteration_dna_seq = dna_seq
    for i in mutable_ind:
      new_picks = picks.copy()
      new_picks[i] = not new_picks[i]
      new_dna_seq = b"".join(words[strand_data,new_picks])
      new_score = penalty_fn(new_dna_seq)
      if new_score < iteration_best_score:
        iteration_best_picks = new_picks
        iteration_dna_seq = new_dna_seq
        iteration_best_score = new_score
    if iteration_dna_seq == dna_seq:
      break
    if iteration_best_score < score:
      picks = iteration_best_picks
      dna_seq = iteration_dna_seq
      score = iteration_best_score
    icount += 1
  logging.debug(f"iteration {icount} picks: {iteration_best_picks}") 

    
  return dna_seq,score

def b32_to_DNA(file_data:list[list[int]],words:list[str], alternate_words:list[str], avoid_seq:list[str] = [])->list[str]:
    """ with avoids, TODO: depricate """
    dna = []
    for strand in file_data:
        _t = [words[x] for x in strand]
        _s = "".join(_t)
        ind,l = find_avoid(_s,avoid_seq)
        _last_s = ""
        while ind>0:
            print("here")
            if ind%3==0:
                rep_index =ind//3
            else:
                rep_index = ind//3+1
            _t[rep_index] = alternate_words[strand[rep_index]]
            _last_s = _s
            _s= "".join(_t)
            if _last_s == _s:
                raise Exception("Can't find solution")
            
            ind,l = find_avoid(_s,avoid_seq)
        dna.append(_s)
    return dna

def dna_to_bN(dna: Union[list[str],list[bytes]] ,words:list[str], alternate_words:list[str])->list[int]:
    """ takes a list of strings and converts them to a list of base N ints"""

    if isinstance(dna[0],bytes):
        dna = [x.decode() for x in dna]

    wordlen = len(words[0])
    #build the reverse lookup table
    bNlut = {}
    for ind in range(len(words)):
        bNlut[words[ind]] = ind
        bNlut[alternate_words[ind]] = ind


    bNdatalist = []
    for d in dna:
        bNdata = []
        for ind in range(0,len(d),wordlen):
            w = d[ind:ind+wordlen] 
            bNdata.append(bNlut[w])
        bNdatalist.append(bNdata)
    return bNdatalist

def dna_to_b32(dna: Union[list[str],list[bytes]] ,words:list[str], alternate_words:list[str])->list[int]:
    assert len(words) == 32
    return dna_to_bN(dna,words,alternate_words)


def __is_pow_two(n: int):
    """Returns if a number is a power of two."""
    return (n > 0) and ((n & ( ~(n-1) )) == n)

def dna_to_bytes(dna_seq: str, words: list[str]=_default_b32_alphabet, alternate_words: list[str]=_default_b32_alphabet) ->  tuple[bytes, list[int]]:
    """Convert a DNA sequence to bytes with zero padding.

    This function takes a DNA sequence and converts it into a byte representation.
    It assumes the alphabet length and base are the same.
    The function ensures that the DNA sequence is properly decoded into bytes, with
    zero padding applied as necessary to align the data.

    Args:
        dna_seq (str): The DNA sequence to be converted.
        words (list[str]): The primary set of base words used, in order, for encoding.
        alternate_words (list[str]): The alternate (aka synonymous) set of words used, in order, for encoding.

    Returns:
        bytes: The byte representation of the DNA sequence.
        list[int]: a mask for alternate words.  0 for word, 1 for alternate word, -1 for don't care.
    """
    if len(words) != len(alternate_words):
        raise ValueError("words and alternate_words must be the same length")
    if len(words) < 2:
        raise ValueError("words and alternate_words must be at least 2 long")
    if len(dna_seq) % len(words[0]) != 0:
        raise ValueError("DNA sequence length must be a multiple of the word length")

    base = len(words)
    word_len = len(words[0])

    #create a mask for the DNA sequence
    mask = [0]*(len(dna_seq)//word_len)
    for i in range(len(mask)):
        word = dna_seq[i*word_len:(i+1)*word_len]
        if word in words:
            mask[i] = 0
        elif word in alternate_words:
            mask[i] = 1
        else:
            raise ValueError("DNA sequence contains invalid word: {}".format(word))

    #convert the DNA sequence to base N
    bN_seq = dna_to_bN([dna_seq], words, alternate_words)[0]
    int_value = _baseN_to_int(bN_seq, base)
    num_words = len(dna_seq)/len(words[0])
    num_bits = num_words * np.log2(base)
    if __is_pow_two(base):
        num_bits = int(np.round(num_bits)) #just incase log2 returns with a rounding error.
    else:
        num_bits = int(np.ceil(num_bits))
    num_bytes = (num_bits + 7) // 8 #only works if num_bits is an int
    
    return int_value.to_bytes(num_bytes, 'little'), mask

def chunk(b:bytes|bytearray,size:int):
    """break into list of size sized chunks"""
    chunks = []
    for i in range(0, len(b), size):
        chunks.append( b[i:i + size] )
    return chunks

def insert_bytes(data:bytes, insert:bytes, chunk_len:int, num_inserts:int) -> bytes:
    """Insert bytes into a byte array num_inserts times at chunk_len intervals.  i.e. 
       the returned bytes will have the insert bytes every chunk_len bytes.

    Args:
        data (bytes): The original byte array.
        insert (bytes): The bytes to be inserted.
        chunk_len (int): The length of each chunk.
        num_inserts (int): The number of insertions to be made.

    Returns:
        bytes: A list of byte chunks with the inserted bytes.

    See also:
        remove_bytes: The inverse of this function.
    """
    data = bytearray(data)
    for i in range(num_inserts):
        ins_index = i * chunk_len
        data[ins_index:ins_index] = insert
    return bytes(data)

def pad_data(data:bytes,block_size:int,padding_data:bytes|str = "random") -> bytes:
    """Pad data to a multiple of block_size.  The padding is added to the end of the data.
    
    Args:
        data (bytes): bytes object to be padded.
        block_size (int): The block size to pad to. (return bytes length % block_size == 0)
        padding_data (bytes|str, optional): The padding data to be added. Defaults to "random".
    Returns:
        bytes: The padded bytes.
    """
    padding_size = block_size - len(data) % block_size 
    if isinstance(padding_data,bytes):
        if len(padding_data) < padding_size:
            raise ValueError("padding data is too small")
    if isinstance(padding_data,str):
        if padding_data == "random":
            padding_data = np.random.bytes(padding_size)
        else:
            raise ValueError("padding data must be bytes or 'random'")
    return data + padding_data[:padding_size]

def remove_bytes(data:bytes, insert:int|bytes, chunk_len:int, num_inserts:int) -> bytes:
    """Remove bytes from a byte array at specified intervals.  This is the inverse of insert_bytes.

    Args:
        data (bytes): The original byte array.
        insert_len (int|bytes): The length of the inserted bytes to be removed.  If insert_len 
            is bytes, its length will be used, and the contents of the deleted bytes will be checked.
        num_inserts (int): The number of insertions to be removed.

    Returns:
        bytes: The modified byte array with the inserted bytes removed.
    """
    if isinstance(insert,bytes):
        insert_len = len(insert)
    else:
        insert = None
    
    data = bytearray(data)
    for i in reversed(range(num_inserts)):
        ins_index = i * chunk_len
        if insert is not None:
            if data[ins_index:ins_index + insert_len] != insert:
                raise ValueError("insert bytes do not match bytes to be removed")
        del data[ins_index:ins_index + insert_len]
    return bytes(data)

