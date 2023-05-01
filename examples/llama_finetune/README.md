This script needs some monkey-patches on the original EasyLM's model definition:

##### Fix Import Errors

EasyLM is based on jax 0.4, while this branch is tested on jax 0.3.22. Some import errors needs to be fixed:

```
--- a/EasyLM/jax_utils.py
+++ b/EasyLM/jax_utils.py
@@ -10,8 +10,8 @@ import dill
 import flax
 import jax
 import jax.numpy as jnp
-from jax.sharding import PartitionSpec as PS
-from jax.sharding import Mesh
+from jax.experimental.pjit import PartitionSpec as PS
+from jax.interpreters.pxla import Mesh
 from jax.experimental.pjit import with_sharding_constraint as _with_sharding_constraint
 from jax.experimental.pjit import pjit
 from jax.interpreters import pxla
```

```
--- a/EasyLM/models/llama/llama_model.py
+++ b/EasyLM/models/llama/llama_model.py
@@ -8,7 +8,7 @@ import numpy as np
 import jax
 import jax.numpy as jnp
 from jax import lax
-from jax.sharding import PartitionSpec as PS
+from jax.experimental.pjit import PartitionSpec as PS
 import flax.linen as nn
 from flax.core.frozen_dict import FrozenDict, freeze, unfreeze
 from flax.linen import combine_masks, make_causal_mask
```

##### Support mark pipeline boundary
We use manual pipeline boundary, though the auto one works in most cases. So we add a marker at the end of each layer.

Will monkey patch it in the training script later.

```
--- a/EasyLM/models/llama/llama_model.py
+++ b/EasyLM/models/llama/llama_model.py
@@ -31,6 +31,7 @@ from mlxu import function_args_to_config, load_pickle, open_file
 from EasyLM.jax_utils import (
     with_sharding_constraint, get_jax_mesh, get_gradient_checkpoint_policy
 )
+from alpa import mark_pipeline_boundary
 
 
 LLAMA_STANDARD_CONFIGS = {
@@ -829,6 +830,7 @@ class FlaxLLaMABlockCollection(nn.Module):
                 output_attentions,
                 fcm_mask,
             )
+            mark_pipeline_boundary()
             hidden_states = layer_outputs[0]
 
             if output_attentions:
```
