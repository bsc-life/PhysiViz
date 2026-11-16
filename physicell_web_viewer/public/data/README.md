# public/data/

Empty on a fresh clone — exported datasets are generated locally, not
shipped with the repo (see the root `.gitignore`). To get something to
look at:

- **Your own PhysiCell/PhysiBoSS run**:
  ```bash
  ./scripts/prepare_dataset.sh /path/to/PhysiCell/output my_run
  ```
- **A quick demo with no custom model**: builds and runs the official,
  unmodified PhysiCell `template` sample project, then exports it —
  needs a C++ toolchain (`g++`, `make`) in addition to what
  `./scripts/install.sh` already sets up:
  ```bash
  ./scripts/demo_from_physicell_sample.sh
  ```

Either way, start the viewer with `./scripts/start.sh` and pick the
dataset from the dropdown (or open `http://localhost:3000/?dataset=<name>`
directly).
