from setuptools import find_packages, setup
from distutils.util import convert_path

with open("README.md") as f:
    readme = f.read()

with open("requirements.txt") as f:
    requirements = f.read().split()

main_ns = {}
ver_path = convert_path('homura/version.py')
with open(ver_path) as ver_file:
      exec(ver_file.read(), main_ns)

setup(name="homura-core",
      version=main_ns['__version__'],
      author="Ryuichiro Hataya",
      author_email="hataya@nlab-mpg.jp",
      packages=find_packages(exclude=["tests", "docs", "examples"]),
      url="https://github.com/moskomule/homura",
      python_requires=">=3.9",
      description="a fast prototyping library for DL research",
      long_description=readme,
      long_description_content_type="text/markdown",
      license="Apache License 2.0",
      install_requires=requirements,
      )
