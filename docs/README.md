# Alpa Documentation

## Build the documentation website

### Dependency
```
pip3 install sphinx sphinx-rtd-theme sphinx-gallery matplotlib
```

### Build
```
make html
```

The build process will execute all tutorial scripts to generate the gallery.
This may cause failures if the build machine does not have necessary environment.
This may also result in a very long build time.
You can set `ALPA_TUTORIAL_EXEC_PATTERN` to only execute the files that match the regular expression pattern.
For example, to build one specific file, do
```
export ALPA_TUTORIAL_EXEC_PATTERN=filename.py
make html
```
To skip execution of all tutorials, do
```
export ALPA_TUTORIAL_EXEC_PATTERN=none
make html
```

### Clean
To remove all generated files:
```
make clean
```

### Serve
Run an HTTP server and visit http://localhost:8000 in your browser.
```
python3 -m http.server --d _build/html
```

### Publish
Clone [alpa-projects.github.io](https://github.com/alpa-projects/alpa-projects.github.io) and make sure you have write access.

```bash
export ALPA_SITE_PATH=~/efs/alpa-projects.github.io   # update this with your path
./publish.py
```

## Add new documentations
Alpa uses [Sphinx](https://www.sphinx-doc.org/en/master/index.html) to generate static documentation website and use [Sphinx-gallery](https://sphinx-gallery.github.io/stable/index.html) to generate gallery examples.

Your new example should be created under `docs/gallery`. 

### Define the Order of Tutorials
You can define the order of tutorials with `subsection_order` and
`within_subsection_order` in [`conf.py`](conf.py).
By default, the tutorials within one subsection are sorted by filename.
