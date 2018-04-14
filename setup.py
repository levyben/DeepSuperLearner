from distutils.core import setup
setup(
    name='DeepSuperLearner',
    version='1.0.0',
    author='Ben Levy',
    description='Implementation of the DeepSuperLearner machine-learning algorithm',
    author_email='levyben1@gmail.com',
    packages=['deepSuperLearner'],
    url='https://github.com/levyben/DeepSuperLearner',
    license='MIT',
    long_description=open('README.md').read(),
    requires=[
        "scipy (>= 1.0.1)",
        "numpy (>= 1.14.2)",
        "sklearn (>= 0.19.1)",
    ],
)
