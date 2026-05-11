# 文件名: setup.py

from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy

# 定义Cython扩展模块
ext_modules = [
    Extension(
        "importance_calculator",  # 最终生成的模块名
        ["importance_calculator.pyx"],  # 源文件名
        include_dirs=[numpy.get_include()]  # 包含Numpy头文件，这对于编译至关重要
    )
]

setup(
    name="Cython Importance Calculator",
    ext_modules=cythonize(ext_modules),
)