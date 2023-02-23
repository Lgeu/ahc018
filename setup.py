from distutils.core import setup, Extension

module = Extension(
    "steiner",
    sources=["steiner.cpp"],
    extra_compile_args=["-O3", "-march=native", "-std=c++17"],
)
setup(
    name="steiner",
    version="0.1.0",
    description="Steiner tree problem solver",
    ext_modules=[module],
)

# python setup.py build_ext --inplace
