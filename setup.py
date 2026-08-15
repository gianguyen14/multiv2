import os
from distutils import log

from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext


class OptionalBuildExt(build_ext):
    """Build the C++ core when possible, but keep Python fallback installable.

    Set UVR_NATIVE_STRICT_BUILD=1 in CI/release builds to turn compiler errors
    into hard failures.
    """

    @property
    def strict(self) -> bool:
        return os.getenv("UVR_NATIVE_STRICT_BUILD", "0").lower() in {"1", "true", "yes"}

    def run(self):
        try:
            super().run()
        except Exception as exc:  # pragma: no cover - compiler/toolchain dependent
            if self.strict:
                raise
            log.warn("optional C++ native core was not built: %s", exc)

    def build_extension(self, ext):
        try:
            super().build_extension(ext)
        except Exception as exc:  # pragma: no cover - compiler/toolchain dependent
            if self.strict:
                raise
            log.warn("optional extension %s was not built: %s", ext.name, exc)


def compile_args():
    if os.name == "nt":
        return ["/O2", "/std:c++17", "/EHsc"]
    return ["-O3", "-std=c++17", "-fvisibility=hidden"]


setup(
    ext_modules=[
        Extension(
            "backend.app.native._core",
            sources=["native/core.cpp"],
            language="c++",
            extra_compile_args=compile_args(),
        )
    ],
    cmdclass={"build_ext": OptionalBuildExt},
)
