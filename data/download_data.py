import os
import gdown


data_dir = os.path.dirname(os.path.abspath(__file__))
object_library_output = os.path.join(data_dir, 'gag/object_library.zip')

print('downloading...')
gdown.download(id='1au3pML1l6qilWkPHRdYi5cKa_mcbINwy', output=object_library_output)
print('finished. extracting files...')
gdown.extractall(object_library_output)
print('done. bye.')
