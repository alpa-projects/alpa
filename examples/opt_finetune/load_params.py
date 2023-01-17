import os
import itertools

import numpy as np
import alpa
from alpa.device_mesh import (DistributedArray, ReplicatedDistributedArray,
                              MeshHostWorker, create_remote_array_refs)
import jax
import flax
from jax.tree_util import tree_flatten, tree_unflatten, tree_leaves
from jax.interpreters import pxla


def load_opt_params_worker_func(self, path, prefix_to_idx, config, shapes,
                                uuids, indices, mesh_ids):
    """The worker function to load OPT parameters."""

    def load_array(key):
        return np.load(os.path.join(path, key))

    def load_param(param_key, loaded_array, is_position_embedding=False):
        i = prefix_to_idx[param_key]

        for j in range(len(mesh_ids[i])):
            if self.mesh_id != mesh_ids[i][j]:
                print(f"skipped because  self.mesh_id {self.mesh_id}, mesh_ids: {mesh_ids[i][j]}")
                continue

            if not is_position_embedding:
                assert shapes[i][j] == loaded_array.shape, (
                    f"{shapes[i][j]} vs. {loaded_array.shape}")
            else:
                if shapes[i][j] != loaded_array.shape:
                    assert shapes[i][j][1] == loaded_array.shape[1]
                    loaded_array = loaded_array[:shapes[i][j][0], :]
            uuid = uuids[i][j]
            datas = []
            for k in range(len(self.local_devices)):
                idx = self.host_id * len(self.local_devices) + k
                datas.append(loaded_array[indices[i][j][idx]])
            self.put_buffers(uuid, datas)

    load_param("model.decoder.embed_tokens.embedding",
               load_array("decoder.embed_tokens.weight"))
    load_param("model.decoder.embed_positions.embedding",
               load_array("decoder.embed_positions.weight"),
               is_position_embedding=True)

    # if config.version > 2:
    load_param("model.decoder.final_layer_norm.scale",
               load_array("decoder.layer_norm.weight"))
    load_param("model.decoder.final_layer_norm.bias",
               load_array("decoder.layer_norm.bias"))

    layers_per_stage = config.num_hidden_layers // config.pp

    for i in range(config.num_hidden_layers):
        if i == 16:
            print(f"stage_id: {i // layers_per_stage}, mesh id: {self.mesh_id}")
        stage_id = i // layers_per_stage
        if stage_id != self.mesh_id:
            continue

        param_prefix = f"model.decoder.layers.{i}."
        load_prefix = f"decoder.layers.{i}."
        # Attention weights
        wq = load_array(load_prefix + "self_attn.q_proj.weight")
        wk = load_array(load_prefix + "self_attn.k_proj.weight")
        wv = load_array(load_prefix + "self_attn.v_proj.weight")
        # dim = wq.shape[-1]
        # w_qkv = np.concatenate([wq, wk, wv], axis=0).reshape(
        #     (3, -1, dim)).transpose([2, 1, 0]).reshape((dim, -1))
        # load_param(param_prefix + "attention.self.qkv_combined.kernel", w_qkv)
        load_param(param_prefix + "self_attn.q_proj.kernel", wq.T)
        load_param(param_prefix + "self_attn.k_proj.kernel", wk.T)
        load_param(param_prefix + "self_attn.v_proj.kernel", wv.T)


        bq = load_array(load_prefix + "self_attn.q_proj.bias")
        bk = load_array(load_prefix + "self_attn.k_proj.bias")
        bv = load_array(load_prefix + "self_attn.v_proj.bias")
        # b_qkv = np.concatenate([bq, bk, bv], axis=0).reshape(
        #     (3, dim)).transpose([1, 0]).reshape((-1,))
        load_param(param_prefix + "self_attn.q_proj.bias", bq)
        load_param(param_prefix + "self_attn.k_proj.bias", bk)
        load_param(param_prefix + "self_attn.v_proj.bias", bv)
        # load_param(param_prefix + "attention.self.qkv_combined.bias", b_qkv)
        load_param(
            param_prefix + "self_attn.out_proj.kernel",
            np.transpose(load_array(load_prefix + "self_attn.out_proj.weight")))
        load_param(param_prefix + "self_attn.out_proj.bias",
                   load_array(load_prefix + "self_attn.out_proj.bias"))
        load_param(param_prefix + "self_attn_layer_norm.scale",
                   load_array(load_prefix + "self_attn_layer_norm.weight"))
        load_param(param_prefix + "self_attn_layer_norm.bias",
                   load_array(load_prefix + "self_attn_layer_norm.bias"))
        # FFN weights
        load_param(param_prefix + "fc1.bias",
                   load_array(load_prefix + "fc1.bias"))
        load_param(param_prefix + "fc1.kernel",
                   np.transpose(load_array(load_prefix + "fc1.weight")))
        load_param(param_prefix + "fc2.bias",
                   load_array(load_prefix + "fc2.bias"))
        load_param(param_prefix + "fc2.kernel",
                   np.transpose(load_array(load_prefix + "fc2.weight")))
        load_param(param_prefix + "final_layer_norm.scale",
                   load_array(load_prefix + "final_layer_norm.weight"))
        load_param(param_prefix + "final_layer_norm.bias",
                   load_array(load_prefix + "final_layer_norm.bias"))


setattr(MeshHostWorker, "load_opt_params_worker_func",
        load_opt_params_worker_func)

def load_params_dis_array(path, executable, params_aval, config, dummy=False):
    """Load parameters with distributed arrays."""
    if dummy:
        alpa.global_config.use_dummy_value_for_benchmarking = True
        params_info, _ = executable.get_input_placement_specs()
        flat_args, in_tree = tree_flatten(params_aval)
        flat_info = tree_leaves(params_info)
        if hasattr(executable, "mesh_group"):
            ret = executable.mesh_group.shard_args_to_arrays(
                flat_info, flat_args)
        else:
            ret = executable.physical_mesh.shard_args_to_arrays_ps(
                flat_info, flat_args)
        alpa.global_config.use_dummy_value_for_benchmarking = False
        return ret

    params_info, _ = executable.get_input_placement_specs()
    params_info = params_info.params

    prefix_to_flat_idx = {}
    ct = itertools.count()

    def dfs(dict_tree, result_dict, cur_prefix):
        if isinstance(dict_tree, (dict, flax.core.FrozenDict)):
            for key in dict_tree.keys():
                dfs(dict_tree[key], result_dict,
                    cur_prefix + ("." if cur_prefix else "") + key)
        else:
            result_dict[cur_prefix] = next(ct)

    dfs(params_aval, prefix_to_flat_idx, "")

    flat_infos, in_tree = tree_flatten(params_info)

    flat_shapes = []
    flat_uuids = []
    flat_indices = []
    flat_mesh_ids = []
    flat_arrays = []

    mesh_group = executable.mesh_group

    for info in flat_infos:
        aval = info.aval
        if len(info.mesh_ids) == 1:
            mesh, spec = mesh_group[info.mesh_ids[0]], info.sharding_specs[0]
            indices = pxla.spec_to_indices(aval.shape, spec)
            ary_refs, ary_uuid = create_remote_array_refs(mesh)
            flat_shapes.append([aval.shape])
            flat_uuids.append([ary_uuid[0]])
            flat_indices.append([indices])
            flat_mesh_ids.append([mesh.mesh_id])
            flat_arrays.append(
                DistributedArray(mesh, aval, spec, ary_refs[0], indices))
        else:
            tmp_shapes = []
            tmp_uuids = []
            tmp_indices = []
            tmp_mesh_ids = []
            tmp_arrays = []
            tmp_meshes = []
            for mesh_id, spec in zip(info.mesh_ids, info.sharding_specs):
                mesh = mesh_group[mesh_id]
                indices = pxla.spec_to_indices(aval.shape, spec)
                ary_refs, ary_uuid = create_remote_array_refs(mesh)
                array = DistributedArray(mesh, aval, spec, ary_refs[0], indices)
                tmp_shapes.append(aval.shape)
                tmp_uuids.append(ary_uuid[0])
                tmp_indices.append(indices)
                tmp_mesh_ids.append(mesh.mesh_id)
                tmp_meshes.append(mesh)
                tmp_arrays.append(array)
            flat_shapes.append(tuple(tmp_shapes))
            flat_uuids.append(tuple(tmp_uuids))
            flat_indices.append(tuple(tmp_indices))
            flat_mesh_ids.append(tuple(tmp_mesh_ids))
            flat_arrays.append(
                ReplicatedDistributedArray(tmp_meshes, tmp_arrays))

    for m in executable.mesh_group.meshes:
        for w in m.workers:
            w.load_opt_params_worker_func.remote(path, prefix_to_flat_idx,
                                                 config, flat_shapes,
                                                 flat_uuids, flat_indices,
                                                 flat_mesh_ids)

    return tree_unflatten(in_tree, flat_arrays)
    # return flat_arrays


def load_multi_executable_params_dis_array(path,
                                           executables,
                                           params_aval,
                                           config,
                                           dummy=False):
    """Load parameters to workers that will be used by all executables. Accordingly,
    we need to make sure the parameter sharding specs are identical for all executables.
    """
    shared_input_shard_specs = None
    # for executable in executables.values():
    #     # stage_input_shard_specs = executable.stage_input_shard_specs
    #     stage_input_shard_specs = executable.get_input_placement_specs
    #     if shared_input_shard_specs is not None:
    #         assert shared_input_shard_specs == stage_input_shard_specs, \
    #             "All executables must have the same input sharding specs."
    #     else:
    #         shared_input_shard_specs = stage_input_shard_specs
    return load_params_dis_array(path,
                                 list(executables.values())[0], params_aval,
                                 config, dummy)
