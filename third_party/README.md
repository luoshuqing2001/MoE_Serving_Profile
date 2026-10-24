Transformers 5.20.0 source, Apache-2.0. The tree at the repo root is gitignored.
Upstream class names trip GitHub secret scanning, so the sdist is downloaded
locally instead of committed.

`record_routing` and `record_gpt_oss` insert `transformers-5.20.0/src` ahead of
the environment package. Run from the repository root.

```bash
pip download transformers==5.20.0 --no-deps --no-binary=:all: -d /tmp/transformers-sdist
tar -xzf /tmp/transformers-sdist/transformers-5.20.0.tar.gz \
    transformers-5.20.0/src transformers-5.20.0/LICENSE transformers-5.20.0/PKG-INFO
```
