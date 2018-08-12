import tensorflow as tf

# constant
a=tf.constant(6.5, name='A')
b=tf.constant(3.4, name='B')
c=tf.constant(3.0, name='C')
d=tf.constant(100.2, name='D')

# a^2
square = tf.square(a,name='square')

# b ^ c
power = tf.pow(b,c,name='pow')

# d ^ -1/2
sqrt = tf.sqrt(d,name='sqrt')

# square + power + sqrt
final_sum = tf.add_n([square,power,sqrt],name='final_sum')

# start session to run tesor network
sess = tf.Session()

print "square of a : " , sess.run(square)
print "power of b^c : ", sess.run(power)
print "square root of d : " , sess.run(sqrt)
print "sum of square, power, sqrt : ", sess.run(final_sum)

# close session
sess.close()
