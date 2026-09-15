# Security and safe use

The runnable release validates dataset shape, identifiers, finite values, labels and input file locations. CIF ZIP members are read without extraction and are limited to 10 MiB each. Checkpoint loading uses PyTorch's restricted `weights_only=True` mode and has no unrestricted pickle fallback. NumPy arrays load with `allow_pickle=False`.

Credentials, automatic cloud uploads and historical broad file-collection utilities are excluded. Notebook outputs are cleared before distribution. New run directories prevent accidental overwrites.

These protections reduce known risks; they do not guarantee that every vulnerability is absent. Crystal parsers and numerical libraries remain dependencies, and hostile files can consume CPU or memory. Process externally supplied structures/checkpoints in an isolated environment with resource limits. Keep dependencies updated and run `python -m pip_audit` in the installed environment.

Historical source snapshots are reference material, not supported executable entry points. They may show old unrestricted deserialization and environment-specific assumptions. Use the `agcgcnn` package and the two documented notebooks for execution.

Report suspected issues privately to the repository owner through GitHub's available private security reporting mechanism. Do not post credentials, private datasets or executable payloads in public issues. Active development is paused, so response times are not guaranteed.
