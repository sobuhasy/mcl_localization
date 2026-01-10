from setuptools import find_packages, setup

package_name = 'mcl_localization'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='sobuhasy',
    maintainer_email='sobuhasy@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'particle_initializer = mcl_localization.particle_initializer:main',
            'mcl_prediction = mcl_localization.mcl_prediction:main',
            'mcl_localization_pf = mcl_localization.mcl_localization_pf:main',
        ],
    },
)
