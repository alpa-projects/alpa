from jax.lib import xla_client

backend = xla_client.get_local_backend("gpu")
g0 = backend.devices()[0]

print(dir(g0))
print(g0.client)
print(g0.device_kind)
print(g0.host_id)
print(g0.id)
print(g0.platform)
print(g0.task_id)

