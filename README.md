Alpa
=======
[**Documentation**](https://alpa-projects.github.io/install/from_source.html) |
[**Slack**](https://join.slack.com/t/alpa-project/shared_invite/zt-13rl45uci-UI~eULQHBHGav4JrwF7dHw)

Alpa automatically parallelizes your python numerical computing code and neural networks
with a simple decorator.

Organization
============
- `examples`: public examples
- `alpa`: the python source code of the library
- `playground`: private experimental scripts
- `tests`: unit tests


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
