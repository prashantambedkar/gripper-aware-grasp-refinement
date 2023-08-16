from setuptools import setup, find_packages
from distutils.extension import Extension
from Cython.Build import cythonize
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import numpy as np
from torch.cuda import is_available as cuda_is_available
import platform


print(f'*******************************************************')
print(f'platform is {platform.system()} and cuda is {"NOT" if not cuda_is_available() else ""} available')
print(f'*******************************************************')

# **** extensions of Convolutional Occupancy Networks

# Get the numpy include directory.
numpy_include_dir = np.get_include()

# mcubes (marching cubes algorithm)
mcubes_module = Extension(
    'convonets.src.utils.libmcubes.mcubes',
    sources=[
        'convonets/src/utils/libmcubes/mcubes.pyx',
        'convonets/src/utils/libmcubes/pywrapper.cpp',
        'convonets/src/utils/libmcubes/marchingcubes.cpp'
    ],
    language='c++',
    extra_compile_args=['-std=c++11'],
    include_dirs=[numpy_include_dir]
)

# triangle hash (efficient mesh intersection)
if platform.system() == 'Windows':
    triangle_hash_module = Extension(
        'convonets.src.utils.libmesh.triangle_hash',
        sources=[
            'convonets/src/utils/libmesh/triangle_hash.pyx'
        ],
        # libraries=['m'],  # Unix-like specific  --> does not work on Windows
        include_dirs=[numpy_include_dir]
    )
else:
    triangle_hash_module = Extension(
        'convonets.src.utils.libmesh.triangle_hash',
        sources=[
            'convonets/src/utils/libmesh/triangle_hash.pyx'
        ],
        libraries=['m'],  # Unix-like specific
        include_dirs=[numpy_include_dir]
    )

# mise (efficient mesh extraction)
mise_module = Extension(
    'convonets.src.utils.libmise.mise',
    sources=[
        'convonets/src/utils/libmise/mise.pyx'
    ],
)

# simplify (efficient mesh simplification)
simplify_mesh_module = Extension(
    'convonets.src.utils.libsimplify.simplify_mesh',
    sources=[
        'convonets/src/utils/libsimplify/simplify_mesh.pyx'
    ],
    include_dirs=[numpy_include_dir]
)

# voxelization (efficient mesh voxelization)
voxelize_module = Extension(
    'convonets.src.utils.libvoxelize.voxelize',
    sources=[
        'convonets/src/utils/libvoxelize/voxelize.pyx'
    ],
    libraries=['m']  # Unix-like specific
)


# **** extensions in gag_refine

if cuda_is_available():
    # earth mover distance
    emd_module = CUDAExtension(
        name='gag_refine.utils.earth_mover_distance.emd_cuda',
        sources=[
            'gag_refine/utils/earth_mover_distance/cuda/emd.cpp',
            'gag_refine/utils/earth_mover_distance/cuda/emd_kernel.cu',
        ],
        extra_compile_args={'cxx': ['-g'], 'nvcc': ['-O2']}
    )

    # chamfer distance
    cd_module = CUDAExtension(
        name='gag_refine.utils.chamfer_distance.cd_cuda',
        sources=[
            'gag_refine/utils/chamfer_distance/chamfer_distance.cpp',
            'gag_refine/utils/chamfer_distance/chamfer_distance_cuda.cu'
        ],
    )

# Gather all extension modules
if platform.system() == 'Windows':
    ext_modules = [
        # mcubes_module,
        triangle_hash_module,
        # mise_module,
        # simplify_mesh_module,
        # voxelize_module,
        # emd_module,       # deactivated for now
        # cd_module,
    ]
else:
    ext_modules = [
        mcubes_module,
        triangle_hash_module,
        mise_module,
        simplify_mesh_module,
        voxelize_module,
        # emd_module,       # deactivated for now
        # cd_module,
    ]

setup(
    name='gripper-aware-grasp-refinement',
    version='1.0',
    ext_modules=cythonize(ext_modules, language_level=3),
    cmdclass={
        'build_ext': BuildExtension
    },
    packages=find_packages(),
    install_requires=[
        'burg_toolkit @ git+https://github.com/mrudorfer/burg-toolkit.git@dev'
    ]
)
