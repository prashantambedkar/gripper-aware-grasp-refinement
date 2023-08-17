"""
This module requires manifold and simplify downloaded from https://github.com/hjwdzh/Manifold and built.
It is assumed that manifold and simplify are in the user PATH variable and can be found.
"""

import os
import pathlib
import subprocess
from tqdm.contrib.concurrent import process_map
import burg_toolkit as burg


def manifold_and_simplify(source_fn, target_fn, n_faces=20000, ratio=0.1):
    """
    This performs the actual manifold and simplify operations:
        `manifold source_fn source_dir/temp.model.obj`
        `simplify -i source_fn -o target_fn -m -c 1e-2 -f [n_faces] -r [ratio]`

    However, manifold/simplify fails for some shapes.

    The method cleans up after itself, i.e. it deletes the target_dir/temp.model.obj (but only if successful).

    :param source_fn: Full path+filename of the source model.obj
    :param target_fn: Full path+filename where to put the target model.obj
    :param n_faces: number of faces when to stop simplifying
    :param ratio: ratio for number of faces when to stop simplifying

    :return: Returns 0 if all went well, returns 1 if failed.
    """
    # manifold - put created file into same directory as source file, but with temp. prefix
    tmp_fn = os.path.join(
        pathlib.Path(source_fn).parent,
        'temp.' + pathlib.Path(source_fn).name
    )
    command = ['manifold', source_fn, tmp_fn]
    cp_m = subprocess.run(command, capture_output=True)
    if cp_m.returncode != 0:
        print(f'manifold failed for {source_fn}')
        # print('manifold stdout', cp_m.stdout)
        # print('manifold stderr', cp_m.stderr)
        return 1

    # simplify
    command = ['simplify', '-i', tmp_fn, '-o', target_fn, '-m', '-c', '1e-2', '-f', str(n_faces), '-r', str(ratio)]
    cp_s = subprocess.run(command, capture_output=True)

    pathlib.Path(tmp_fn).unlink(missing_ok=True)  # clean up temporary file

    if cp_s.returncode != 0:
        print(f'simplify failed for {source_fn}')
        # print('manifold stdout', cp_m.stdout)
        # print('manifold stderr', cp_m.stderr)
        # print('simplify stdout', cp_s.stdout)
        # print('simplify stderr', cp_s.stderr)
        return 1

    print(f'finished manifold and simplify for {pathlib.Path(target_fn).name}')
    return 0


def watertight_helper(source_and_target_filenames):
    # required for parallelisation (a single iterable argument)
    source_fn, target_fn = source_and_target_filenames
    manifold_and_simplify(source_fn, target_fn)


def make_meshes_watertight(object_library, target_dir='meshes'):
    full_target_dir = pathlib.Path(object_library.filename).parent / target_dir
    burg.io.make_sure_directory_exists(full_target_dir)

    source_and_target_filenames = []
    identifiers = []
    for identifier, obj in object_library.items():
        source_fn = obj.mesh_fn
        target_fn = full_target_dir / f'{identifier}.obj'
        source_and_target_filenames.append((source_fn, target_fn))
        identifiers.append(identifier)

    results = process_map(watertight_helper, source_and_target_filenames, chunksize=10)

    errored = []
    for ret_val, files, identifier in zip(results, source_and_target_filenames, identifiers):
        if ret_val == 0:
            object_library[identifier].mesh_fn = files[1]
        else:
            errored.append(object_library.pop(identifier))

    print(f'removed {len(errored)} objects from library, as they could not be made watertight. they are:')
    for element in errored:
        print(f'{element.identifier} - {element.name}')

    object_library.to_yaml(object_library.filename + '_new')
    print(f'new object library saved with suffix _new')
