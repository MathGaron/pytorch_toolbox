from distutils.core import setup

setup(
    name='pytorch_toolbox',
    version='0.1dev',
    packages=['pytorch_toolbox',
              'pytorch_toolbox.network',
              'pytorch_toolbox.loader',
              'pytorch_toolbox.visualization',
              'pytorch_toolbox.transformations'], requires=['numpy']
)
