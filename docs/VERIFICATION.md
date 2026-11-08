# Repository verification

Packaging verification on 2026-11-08:

- The nine V2 scientific-contract unit tests passed in the existing Python 3.11 environment.
- The CLI help command completed successfully. All included Python files parsed successfully.
- The preserved V2 source digest exactly matches the formal run record: `b42ddf1964ef1b200f5a6715d2ce5bb9f3e68750b6f08296e623de0d742246f7`.
- Sealed run, freeze and V2 dataset records validated. All five included files referenced by the freeze manifest matched their recorded SHA256 values. Omitted NPZ indices were not checked through the repository snapshot.
- The existing formal reload report records all 26 final/shift models as passing. That is historical experiment evidence, not a new model-inference run during packaging.
- Selected text files had no matches for common credential/key patterns or user-home paths. This is a limited pattern scan, not a security audit.
- Raw data, processed arrays, model artifacts, logs, virtual environments and full predictions are excluded from Git.

Full dataset download, fresh training, fresh inference and cross-device reproducibility were not rerun while packaging. No CI workflow has been configured. The original completed experiment was left untouched.
