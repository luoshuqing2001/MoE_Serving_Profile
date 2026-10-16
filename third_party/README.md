Transformers 5.19.0 source, Apache-2.0. This directory is gitignored.
Upstream class names trip GitHub secret scanning, so the sdist is downloaded
locally instead of committed.

`record_routing` inserts `transformers-5.19.0/src` ahead of the environment
package so Qwen3.8-Flash-Next FP8 routing uses this tree.

```bash
pip download transformers==5.19.0 --no-deps --no-binary=:all: -d /tmp/transformers-sdist
tar -xzf /tmp/transformers-sdist/transformers-5.19.0.tar.gz \
    -C third_party transformers-5.19.0/src transformers-5.19.0/LICENSE transformers-5.19.0/PKG-INFO
```
