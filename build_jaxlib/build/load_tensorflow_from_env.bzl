def _load_tensorflow_from_env_impl(ctx):
    tf_path = ctx.os.environ['TF_PATH']
    ctx.symlink(tf_path, "")

load_tensorflow_from_env = repository_rule(
    implementation = _load_tensorflow_from_env_impl,
    local=True,
    environ=["TF_PATH"],
)
