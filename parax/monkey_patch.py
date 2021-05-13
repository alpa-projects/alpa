"""Monkey patch other python libraries."""

#import flax

# DEPRECATED!
# Patch __eq__ and __hash__ function for OptimizerDef in flax

#def __hash__(self):
#    return hash(self.hyper_params.learning_rate)
#def __eq__(self, other):
#    return True
#setattr(flax.optim.GradientDescent, "__hash__", __hash__)
#setattr(flax.optim.GradientDescent, "__eq__", __eq__)
