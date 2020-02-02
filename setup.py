from setuptools import setup

setup(
    # Needed to silence warnings (and to be a worthwhile package)
    name='grad_cam',
    # author='some author',
    # author_email='',
    # Needed to actually package something
    packages=['grad_cam','keras_pkg'],
    # install_requires=['numpy'], # Not required here
    #version='0.1',
    description='grad cam'
    )
