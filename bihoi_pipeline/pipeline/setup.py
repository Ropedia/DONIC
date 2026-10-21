from setuptools import setup, find_packages
setup(name='hoi-easymocap', version='0.2.1', packages=find_packages(include=['easymocap', 'easymocap.*']), package_data={'easymocap':['**/*.txt','**/*.yml','**/*.yaml']})
