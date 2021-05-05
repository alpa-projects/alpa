from jax.core import Primitive


p0 = Primitive("pipeline")
p1 = Primitive("pipeline")

print(p0.__hash__())
print(p1.__hash__())
print(p0 == p1)
