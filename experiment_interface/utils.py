import numpy as np

def np_encode(x):
	'''
	Input
	=====
		x : np.ndarray


	Return
	======
		encoded_array: bytes
		encoded_shape: bytes
		encoded_dtype: bytes
	'''

	encoded_array = x.tostring()
	encoded_shape = str(x.shape)[1:-1]
	encoded_dtype = str(x.dtype)

	return encoded_array, encoded_shape, encoded_dtype


def np_decode(encoded_array, encoded_shape, encoded_dtype):           

	decoded_dtype = np.dtype(encoded_dtype)
	decoded_shape = tuple( np.fromstring(encoded_shape, dtype=np.int64, sep=',') )
	# decoded_array = np.fromstring(encoded_array, dtype=decoded_dtype).reshape(decoded_shape)
	decoded_array = np.frombuffer(encoded_array, dtype=decoded_dtype).reshape(decoded_shape)

	return decoded_array

def extract_np_array(df, name):
	'''
	Decode numpy array from DataFrame

	Input
	=====
		df: pd.DataFrame
		name: str
			df must have <name>, <name>_shape, <name>_dtype as columns.

	Return
	======
		decoded: pd.Series
			Each element is a np.array.

	'''
	_df = df[ [name, name+'_shape', name+'_dtype'] ]
	decoded = df.apply(lambda x: np_decode(x[name], x[name+'_shape'], x[name+'_dtype']), axis=1)
	return decoded


