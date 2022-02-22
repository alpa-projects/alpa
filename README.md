Alpa
=======
[**Documentation**](https://alpa-projects.github.io) |
[**Slack**](https://forms.gle/YEZTCrtZD6EAVNBQ7)

Alpa automatically parallelizes tensor computational graphs and runs them on a distributed cluster. 

Organization
============
- This Repo
  - `alpa`: the python source code of Alpa
  - `benchmark`: benchmark scripts
  - `docs`: documentation and tutorials
  - `examples`: public examples
  - `playground`: private experimental scripts
  - `tests`: unit tests

- [tensorflow-alpa](https://github.com/alpa-projects/tensorflow-alpa). The tensorflow fork for Alpa.
  The c++ source code of Alpa mainly resides in `tensorflow/compiler/xla/service/spmd`.

- [jax-alpa](https://github.com/alpa-projects/jax-alpa). The jax fork for Alpa.
  We do not change any functionatiy, but modify the building scripts to make building with tensorflow-alpa easier.

Formatting & Linting
============
Install prospector and yapf via:
```bash
pip install prospector yapf
```

Use yapf to automatically format the code:
```bash
./format.sh
```

Then use prospector to run linting for the folder ``alpa/``:
```bash
prospector alpa/
```

Style guidelines:
- We follow Google Python Style Guide: https://google.github.io/styleguide/pyguide.html.
- **Avoid using backslash line continuation as much as possible.** yapf will not format well lines with backslash line continuation.
  Make use of [Python’s implicit line joining inside parentheses, brackets and braces.](http://docs.python.org/reference/lexical_analysis.html#implicit-line-joining)
  If necessary, you can add an extra pair of parentheses around an expression.
