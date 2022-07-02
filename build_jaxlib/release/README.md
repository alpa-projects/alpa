# How to Release JaxLib and generate a PyPI Index

## Upload jaxlib wheels as assets of a release tag
```shell
GITHUB_TOKEN="admin_token" python wheel_upload.py --tag [TAG] --path [PATH_TO_WHEELS]
```

## Generate a html index page and commit it to Alpa doc page
```shell
python generate_pypi_index.py --tag [TAG]
```
All wheel assets under `[TAG]` will be included in a html index page appeared in the doc repo.
