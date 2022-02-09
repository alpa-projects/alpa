# Alpa Documentation

## Build the documentation website

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
Clone [alpa-projects.github.io](https://github.com/alpa-projects/alpa-projects.github.io) and make sure you have write access.

```bash
export ALPA_SITE_PATH=~/efs/alpa-projects.github.io   # update this with your path
./publish.py
```

## Add new documentations
Alpa uses [Sphinx](https://www.sphinx-doc.org/en/master/index.html) to generate static documentation website and use [Sphinx-gallery](https://sphinx-gallery.github.io/stable/index.html) to generate gallery examples.

Your new example should be created under `docs/gallery`. 
