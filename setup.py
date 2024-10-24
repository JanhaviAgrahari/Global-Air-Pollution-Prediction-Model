from setuptools import find_packages, setup
from typing import List

HYPEN_E_DOT = "-e ."


def get_requirements(file_path: str) -> List[str]:
    """this function will return the list of requirments"""
    requirements = []
    with open(file_path) as file_obj:
        requirements = file_obj.readlines()
        requirements = [req.replace("\n", "") for req in requirements]
        if HYPEN_E_DOT in requirements:
            requirements.remove(HYPEN_E_DOT)

    return requirements


setup(
    name="Global Air Pollution Prediction Model",
    version="0.0.1",
    author="Janhavi",
    author_email="janhaviagrahari1410@gmail.com",
    packages=find_packages(),  # Fix here: remove the extra comma
    install_requires=get_requirements("requirements.txt"),
)