"""Generate and upload a PyPI index page given a tag."""
import os
import logging
import argparse
import subprocess
from datetime import datetime

import github3
import github3.session as session
import requests


def py_str(cstr):
    return cstr.decode("utf-8")


def url_is_valid(url):
    """Check if a given URL is valid, i.e. it returns 200 OK when requested."""
    r = requests.get(url)

    if r.status_code != 200:
        print("Warning: HTTP code %s for url %s" % (r.status_code, url))

    return r.status_code == 200


def list_wheels(repo, tag):
    gh = github3.GitHub(token=os.environ["GITHUB_TOKEN"],
                        session=session.GitHubSession(default_connect_timeout=100, default_read_timeout=100))
    repo = gh.repository(*repo.split("/"))
    wheels = []
    all_tags = [release.tag_name for release in repo.releases()]
    if tag not in all_tags:
        raise RuntimeError("The tag provided does not exist.")
    release = repo.release_from_tag(tag)
    for asset in release.assets():
        print(f"Validating {asset.name} with url: {asset.browser_download_url}")
        if asset.name.endswith(".whl") and url_is_valid(asset.browser_download_url):
            wheels.append(asset)
    return wheels


def update_wheel_page(keep_list, site_repo, tag, dry_run=False):
    """Update the wheel page"""
    new_html = ""
    for asset in keep_list:
        new_html += '<a href="%s">%s</a><br>\n' % (
            asset.browser_download_url,
            asset.name,
        )

    def run_cmd(cmd):
        proc = subprocess.Popen(
            cmd, cwd=site_repo, stdout=subprocess.PIPE, stderr=subprocess.STDOUT
        )
        (out, _) = proc.communicate()
        if proc.returncode != 0:
            msg = "git error: %s" % cmd
            msg += py_str(out)
            raise RuntimeError(msg)

    run_cmd(["git", "fetch"])
    run_cmd(["git", "checkout", "-B", "master", "origin/master"])
    wheel_html_path = os.path.join(site_repo, "wheels.html")
    if not os.path.exists(wheel_html_path) or open(wheel_html_path, "r").read() != new_html:
        print(f"Wheel page changed, update {wheel_html_path}..")
        if not dry_run:
            open(wheel_html_path, "w").write(new_html)
            run_cmd(["git", "add", "wheels.html"])
            run_cmd(["git", "commit", "-am",
                     f"wheel update at {datetime.now()} from tag {tag}"])
            run_cmd(["git", "push", "origin", "master"])


def delete_assets(remove_list, dry_run):
    for asset in remove_list:
        if not dry_run:
            asset.delete()
    if remove_list:
        print("Finish deleting %d removed assets" % len(remove_list))


def main():
    logging.basicConfig(level=logging.WARNING)
    parser = argparse.ArgumentParser(
        description="Generate a wheel page given a release tag, assuming the wheels have been uploaded."
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--site-path", type=str, default="alpa-projects.github.io")
    parser.add_argument("--repo", type=str, default="alpa-projects/alpa")
    parser.add_argument("--tag", type=str)

    if "GITHUB_TOKEN" not in os.environ:
        raise RuntimeError("need GITHUB_TOKEN")
    args = parser.parse_args()
    wheels = list_wheels(args.repo, args.tag)
    update_wheel_page(wheels, args.site_path, args.tag, args.dry_run)


if __name__ == "__main__":
    main()
