"""Update the wheels page, prune old nightly builds if necessary."""
import github3
import os
import logging
import argparse
import subprocess
from datetime import datetime

import requests


def py_str(cstr):
    return cstr.decode("utf-8")


def extract_group_key_order(name):
    """Extract group key and order from name.

    Parameters
    ----------
    name : str
        name of the file.

    Returns
    -------
    group_key : tuple
        The group the build should belong to

    order : tuple
        The order used to sort the builds.
        The higher the latest
    """
    assert name.endswith(".whl")
    name = name[:-4]
    arr = name.split("-")

    pkg_name = arr[0]
    group_key = [arr[0]] + arr[2:]

    ver = arr[1]
    plus_pos = ver.find("+")
    if plus_pos != -1:
        ver = ver[:plus_pos]

    dev_pos = ver.find(".dev")
    if dev_pos != -1:
        # all nightly share the same group
        group_key.append("nightly")
        # dev number as the order.
        pub_ver = [int(x) for x in ver[:dev_pos].split(".")]
        order = pub_ver + [int(ver[dev_pos + 4 :])]
    else:
        # stable version has its own group
        group_key.append(ver)
        order = [0]

    return tuple(group_key), tuple(order)


def group_wheels(wheels):
    group_map = {}
    for asset in wheels:
        gkey, order = extract_group_key_order(asset.name)
        if gkey not in group_map:
            group_map[gkey] = []
        group_map[gkey].append((order, asset))
    return group_map


def url_is_valid(url):
    """Check if a given URL is valid, i.e. it returns 200 OK when requested."""
    r = requests.get(url)

    if r.status_code != 200:
        print("Warning: HTTP code %s for url %s" % (r.status_code, url))

    return r.status_code == 200


def run_prune(args, group_map):
    keep_list = []
    remove_list = []
    for key, assets in group_map.items():
        print(f"Group {key}:")
        for idx, item in enumerate(reversed(sorted(assets, key=lambda x: x[0]))):
            order, asset = item
            if idx < args.keep_top:
                print("keep  %s" % asset.browser_download_url)
                keep_list.append(asset)
            else:
                print("remove  %s" % asset.browser_download_url)
                remove_list.append(asset)
        print()
    return keep_list, remove_list


def list_wheels(repo):
    gh = github3.login(token=os.environ["GITHUB_TOKEN"])
    repo = gh.repository(*repo.split("/"))
    wheels = []

    for release in repo.releases():
        tag = release.tag_name
        for asset in release.assets():
            if asset.name.endswith(".whl") and url_is_valid(asset.browser_download_url):
                wheels.append(asset)
    return wheels


def update_wheel_page(keep_list, site_repo, dry_run=False):
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
    run_cmd(["git", "checkout", "-B", "main", "origin/main"])
    wheel_html_path = os.path.join(site_repo, "wheels.html")
    if open(wheel_html_path, "r").read() != new_html:
        print(f"Wheel page changed, update {wheel_html_path}..")
        if not dry_run:
            open(wheel_html_path, "w").write(new_html)
            run_cmd(["git", "commit", "-am", "wheel update at %s" % datetime.now()])
            run_cmd(["git", "push", "origin", "main"])


def delete_assets(remove_list, dry_run):
    for asset in remove_list:
        if not dry_run:
            asset.delete()
    if remove_list:
        print("Finish deleting %d removed assets" % len(remove_list))


def main():
    logging.basicConfig(level=logging.WARNING)
    parser = argparse.ArgumentParser(
        description="Prune nightly build and synchronize the wheel page."
    )
    parser.add_argument("--keep-top", type=int, default=1)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--site-path", type=str, default="tlc-pack.github.io")
    parser.add_argument("--repo", type=str, default="tlc-pack/tlcpack")

    if "GITHUB_TOKEN" not in os.environ:
        raise RuntimeError("need GITHUB_TOKEN")
    args = parser.parse_args()
    wheels = list_wheels(args.repo)
    group_map = group_wheels(wheels)
    keep_list, remove_list = run_prune(args, group_map)
    # NOTE: important to update html first before deletion
    # so that the wheel page always points to correct asset
    update_wheel_page(keep_list, args.site_path, args.dry_run)
    delete_assets(remove_list, args.dry_run)


if __name__ == "__main__":
    main()
