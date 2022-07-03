# How to Release JaxLib and generate a PyPI Index

1.  Upload jaxlib wheels as assets under a release tag.
```shell
GITHUB_TOKEN=[ADMIN_TOKEN] python wheel_upload.py --tag [TAG] --path [PATH_TO_WHEELS]
```

2. Generate a html index page and commit it to the master branch of Alpa doc repository.
```shell
GITHUB_TOKEN=[ADMIN_TOKEN] python generate_pypi_index.py --tag [TAG]
```
All wheel assets under `[TAG]` will be included in a html index page appeared in the doc repo.

Please make sure the TAG is aligned in Step 1 and Step 2.
