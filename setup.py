# setup.py file will be responsible for creating a ML or DL application as a package. With the help of this file can create the application as a package and even deploy in PyPy. From there anybody can do the installation and anybody can use it.

from setuptools import find_packages, setup # will find all the packages in the application
from typing import List # For returning a list

HYPEN_E_DOT = '-e .'
def get_requirements(file_path:str)->List[str]:
    '''
    This function will return the list of requirements

    '''
    requirements=[]
    with open(file_path) as file_obj: # file_path=requirements.txt
        requirements=file_obj.readlines()
        requirements=[req.replace("\n", "") for req in requirements]

        if HYPEN_E_DOT in requirements:
            requirements.remove(HYPEN_E_DOT)
    return requirements


setup(
    name= 'dlproject',
    version= '0.0.1',
    author='Madushani',
    author_email='mmweerasekara@gmail.com',
    packages=find_packages(),
    install_requires=get_requirements('requirements.txt')

)