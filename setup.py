from setuptools import setup, find_packages
 
setup(
    name="NUBot",
    version="0.1.0",
    packages=find_packages(include=["src*", "src.*"]),
    include_package_data=True,
    #classifiers=[
        #"Programming Language :: Python :: 3",
        #"License :: OSI Approved :: MIT License",
        #"Operating System :: OS Independent",
    #],
    python_requires='>=3.8',
)
 