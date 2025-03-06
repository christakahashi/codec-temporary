#write a test for codec.py

import random
import numpy as np

import codec

#logging.basicConfig(level=logging.DEBUG,stream=sys.stdout)
#numba_logger = logging.getLogger('numba')
#numba_logger.setLevel(logging.WARNING)

def test_encode_decode(): 
  c = codec.BaseNBlockCodec(inner_alphabet_size=32,inner_d=5,inner_n=30)

  in_text = (codec.lipsum + codec.lipsum)[0:c.block_capacity_bytes]

  coded = c.encode( in_text )
  random.shuffle(coded)
  #coded = coded[:-4] #erase a random strands
  corrupt_index = random.randint(0,len(coded[0])) 
  coded[0][corrupt_index] = 6 

  out_text = c.decode( coded ) 
  assert in_text == out_text

def test_coded_to_bases():
  #TODO: test optimize too

  test_encoded_data = np.random.randint(0,32,(10,31))
  DNA_with_scores = codec.b32_to_DNA_optimize(test_encoded_data,codec._default_b32_alphabet,codec._default_b32_alphabet_alt)
  DNA = [x[0] for x in DNA_with_scores]
  test_decoded_data  = codec.dna_to_b32(DNA,codec._default_b32_alphabet,codec._default_b32_alphabet_alt) 
  print(test_encoded_data.flatten())
  print( np.array(test_decoded_data).flatten())
  assert np.array_equal( test_encoded_data.flatten(), np.array(test_decoded_data).flatten() )
