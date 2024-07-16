from setuptools import setup, find_packages
from typing import List

HYPEN_E_DOT='-e .'
def get_requirements(file_path:str)->List[str]:
    '''
    this function will return the list of requirements; takes file path as argument
    '''
    requirements=[]
    with open(file_path) as file_obj:
        requirements=file_obj.readlines()
        requirements=[req.replace("\n","") for req in requirements]

        if HYPEN_E_DOT in requirements:
            requirements.remove(HYPEN_E_DOT)
    
    return requirements

setup(
    name='Predict-House-Prices-in-India',
    version='0.1',
    packages=find_packages(),
    install_requires=get_requirements('requirements.txt'),
    author='Paridhi',
    description='Predict House Prices in India',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    license='MIT',
    keywords='Predict House Prices in India machine learning regression random forest decision tree catboost',
)
