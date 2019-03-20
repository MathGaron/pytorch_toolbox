from distutils.core import setup

setup(
    name='pytorch_toolbox',
    version='0.1dev',
    packages=['pytorch_toolbox',
              'pytorch_toolbox.visualization',
              'pytorch_toolbox.transformations',
              'pytorch_toolbox.modules'],
    requires=['numpy', 'tqdm', 'visdom', 'pyyaml', ]
)
