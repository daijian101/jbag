import setuptools

setuptools.setup(
    name='jbag',
    version='4.1.0',
    author='Dai Jian',
    author_email='daijian@stumail.ysu.edu.cn',
    description='Tools for medical image processing and deap learning.',
    long_description_content_type='text/markdown',
    packages=setuptools.find_packages(),
    classifiers=[
        'Programming Language :: Python :: 3.11',
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
        'Operating System :: OS Independent',
    ],
    python_requires='~=3.11',
    install_requires=[
        'numpy >= 1.21.0',
        'nibabel >= 4.0.1',
        'scipy >= 1.7.3',
        'SimpleITK >= 2.1.1.2',
        'pydicom >= 2.4.3',
        'dicom2nifti >= 2.4.8',
        'pandas',
        'openpyxl',
        'python-docx',
        'scikit-image',
    ]
)
