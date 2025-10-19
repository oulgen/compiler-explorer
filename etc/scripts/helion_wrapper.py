# Copyright (c) 2025, Compiler Explorer Authors
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
#     * Redistributions of source code must retain the above copyright notice,
#       this list of conditions and the following disclaimer.
#     * Redistributions in binary form must reproduce the above copyright
#       notice, this list of conditions and the following disclaimer in the
#       documentation and/or other materials provided with the distribution.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

from __future__ import annotations

import argparse
import contextlib
import importlib.util
import inspect
import sys
from types import ModuleType


def _load_module(*, path: str, name: str = "example") -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _make_fake_args(kernel) -> tuple[object, ...]:
    try:
        import torch  # type: ignore
    except Exception:
        # Torch is required by Helion; if unavailable, return no args to trigger a clean skip
        return tuple()

    sig = inspect.signature(kernel.fn)
    fake_args = []
    for param in sig.parameters.values():
        ann = param.annotation
        # Prefer obvious types; default to a Tensor to satisfy device discovery
        try:
            if ann is int:
                fake_args.append(1)
            elif ann is float:
                fake_args.append(1.0)
            elif ann is bool:
                fake_args.append(False)
            else:
                fake_args.append(torch.empty([1]))
        except Exception:
            fake_args.append(None)
    return tuple(fake_args)


def main() -> None:
    parser = argparse.ArgumentParser(description="Output Triton code from public Helion kernels.")
    parser.add_argument("--inputfile", required=True)
    parser.add_argument("--outputfile", required=True)
    args = parser.parse_args()

    try:
        mod = _load_module(path=args.inputfile)
        # Import helion lazily after module load in case the file sets env vars
        import helion as _hl
        from helion.runtime.kernel import Kernel

        # Collect Helion kernels (values that are instances of Kernel and are public)
        kernels: list[Kernel] = [
            value
            for name, value in inspect.getmembers(mod)
            if not name.startswith("_") and isinstance(value, Kernel)
        ]

        # For deterministic output, sort by source line numbers if possible
        def _line_no(k: Kernel) -> int:
            try:
                return k.fn.__code__.co_firstlineno  # type: ignore[attr-defined]
            except Exception:
                return 0

        kernels = sorted(set(kernels), key=_line_no)

        # If no kernels found, do nothing (empty output)
        with open(args.outputfile, "w", encoding="utf-8") as out:
            for kernel in kernels:
                try:
                    bound = kernel.bind(_make_fake_args(kernel))
                    # Use a deterministic default config to avoid autotuning
                    cfg = bound.config_spec.default_config()
                    triton_code = bound.to_triton_code(cfg)
                except Exception:
                    # Best effort: fallback to printing a user hint
                    continue
                out.write(triton_code)
                out.write("\n\n")

    except Exception as error:
        # Match CE Python wrapper semantics on error reporting & exit code
        messages = [m for m in (getattr(error, "args", None) or [str(error)])]
        with contextlib.suppress(Exception):
            sys.stderr.writelines([str(m) + "\n" for m in messages])
        sys.exit(255)


if __name__ == "__main__":
    main()


