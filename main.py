# import tensorflow
import tensorflow  as tf

# create some sample tensors of different data types
string = tf.Variable("this is a string", tf.string)
number = tf.Variable(324, tf.int16)
floating = tf.Variable(3.1415926535, tf.float64)

# create different rank tensors
rank0_tensor = tf.Variable("this is a string", tf.string)
rank1_tensor = tf.Variable(["yes", "no"], tf.string)
rank2_tensor = tf.Variable([["yes", "no"], ["yes", "no"]], tf.string)
rank3_tensor = tf.Variable([[["yes", "no"], ["yes", "no"],["yes", "no"]],[["yes", "no"], ["yes", "no"],["yes", "no"]]], tf.string)

# use the rank function to output the rank details
tensor_to_check = rank3_tensor
print(f'The inputted tensor has a rank of {tf.rank(tensor_to_check).numpy()}')

# use the shape attribute to get shape details
print(f'The inputted tensor has a shape of {tensor_to_check.shape}')

# reshape the rank3_tensor 
tensor1 = tf.reshape(rank3_tensor, [2, -1])
print(tensor1)

# evaluate tensor1
print(tensor1.numpy())