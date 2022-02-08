# Alpa Documentation

## Build
```
pip3 install sphinx sphinx-rtd-theme
make html
```

## Serve
Run an HTTP server and visit http://localhost:8000 in your browser.
```
python3 -m http.server --d _build/html
```

## Publish
```
export ALPA_SITE_PATH=~/efs/alpa-projects.github.io   # update this with your path
cp -r _build/html/* $ALPA_SITE_PATH
```

Commit the change and push to the master branch of alpa-projects.github.io.
