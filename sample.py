import tensorflow as tf
a=tf.constant(6,name="A")
b=tf.constant(3,name="B")
c=tf.constant(10,name="C")
d=tf.constant(5,name="D")
mul=tf.multiply(a,b,name="mul")
div=tf.div(c,d,name="div")
addn=tf.add_n([a,b,c,d,mul,div],name="add_n")
print addn

sess=tf.Session()
print sess.run(addn)

print sess.run(div)

print sess.run(mul)

sess.close();

