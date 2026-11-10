"""Checks of a case's segmentation files against their manifest (<case>.shell.json). Stdlib only."""

import hashlib


class ManifestMismatch(ValueError):
    pass


def file_sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def check_manifest(manifest: dict, hashes: dict, class_labels: dict, shell_values=None, label_values=None) -> None:
    """Raise ManifestMismatch unless the manifest matches the files and the protocol.

    hashes        {"shell": sha256, "labels": sha256} of the files on disk
    class_labels  the protocol's class name -> label value
    shell_values, label_values
                  the nonzero values in the shell and filled label maps (optional: checked when given)
    """
    want = manifest.get("sha256") or {}
    for key in ("shell", "labels"):
        if want.get(key) != hashes.get(key):
            raise ManifestMismatch(f"the {key} file's sha256 does not match the manifest: rebuild the shells")
    classes = manifest.get("classes") or []
    unknown = [c for c in classes if c not in class_labels]
    if unknown:
        raise ManifestMismatch(f"the manifest names classes {unknown} that the profile does not have")
    labels = manifest.get("labels") or {}
    wrong = {c: labels.get(c) for c in classes if labels.get(c) != class_labels[c]}
    if wrong:
        raise ManifestMismatch(f"the manifest's labels {wrong} differ from the profile's: rebuild the shells")
    expected = sorted(class_labels[c] for c in classes)
    for key, values in (("shell", shell_values), ("labels", label_values)):
        if values is None:
            continue
        got = sorted(int(v) for v in values if int(v))
        if got != expected:
            raise ManifestMismatch(f"the {key} file holds labels {got}, the manifest {expected}: rebuild the shells")
