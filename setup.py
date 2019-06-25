import os
from setuptools import setup

setup(name='Compare Tools',
      packages=["compare_tools"],
      version='0.0.1',
      description="Compare HathiTrust Books.",
      url="https://github.com/massivetexts/compare-tools",
      author="Peter Organisciak",
      author_email="peter.organisciak@du.edu",
      license="MIT",
      classifiers=[
        'Intended Audience :: Education',
        "Natural Language :: English",
        'License :: OSI Approved :: MIT License',
        "Operating System :: Unix",
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.1',
        "Topic :: Text Processing :: Indexing",
        "Topic :: Text Processing :: Linguistic"
        ],
      install_requires=["numpy", "pandas"]
)
