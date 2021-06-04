import setuptools

# set the version here
version = '0.1.0a'

with open("README.md", "r") as fh:
    long_description = fh.read()
setuptools.setup(
    name="clip",
    version=version,
    author="",
    author_email="",
    description="OpenAI clip in paddle",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="",
    packages=setuptools.find_packages(
        exclude=["build*", "test*", "examples*"]),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
    install_requires=open('requirements.txt').read().split('\n'),
)
