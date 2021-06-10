from setuptools import setup, find_packages
import pathlib

file_dir = pathlib.Path(__file__).parent.resolve()

long_description = (file_dir / 'README.md').read_text(encoding='utf-8')

setup(
    name='vector_cv_tools',
    version='0.1.0',
    description='Vector Institute Compute Vision Project Tools',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/VectorInstitute/vector_cv_tools',
    author='Xin Li, Gerald Shen',
    author_email='xin.li@vectorinstitute.ai,gerald.shen@vectorinstitute.ai',
    classifiers=[
        'Programming Language :: Python :: 3',
    ],
    packages=find_packages(),
    python_requires='>=3.6',
    entry_points={
        'console_scripts': [
            'create_flow=vector_cv_tools.preprocessing.create_flow:main'
        ],
    },
    install_requires=[
        'albumentations',
        'pycocotools',
        'torch',
        'torchvision',
        'tqdm',
        'scikit-image',
    ],
)
