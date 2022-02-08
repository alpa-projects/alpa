# Alpa Documentation

## Build docs

### Dependency
```
pip3 install sphinx sphinx-rtd-theme sphinx-gallery
```

### Build
```
make html
```

### Serve
Run an HTTP server and visit http://localhost:8000 in your browser.
```
python3 -m http.server --d _build/html
```

### Publish
```
export ALPA_SITE_PATH=~/efs/alpa-projects.github.io   # update this with your path
cp -r _build/html/* $ALPA_SITE_PATH
```

Commit the change and push to the master branch of alpa-projects.github.io.


## Add new documentations
Alpa uses [Sphinx](https://www.sphinx-doc.org/en/master/index.html) to generate static documentation website and use [Sphinx-gallery](https://sphinx-gallery.github.io/stable/index.html) to generate gallery examples.

Your new example should be created under `docs/gallery`. 

