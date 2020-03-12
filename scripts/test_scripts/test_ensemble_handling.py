from tqdm import tqdm
from morphx.data.chunkhandler import ChunkHandler
from morphx.processing import clouds, objects

print("Starting now...")

data_path = '/u/jklimesch/thesis/gt/gt_ensembles/'
chunk_size = 15000
npoints = 5000

ch = ChunkHandler(data_path, chunk_size, npoints, specific=False)

total = None
for ix in tqdm(range(len(ch))):
    chunk = ch[0]
    if total is None:
        total = chunk
    else:
        total = clouds.merge_clouds([total, chunk])
objects.save2pkl(total, data_path + 'test_chunks_new/', name='total')


# ch = ChunkHandler(data_path, chunk_size, npoints, specific=True)
#
# for obj in ch.obj_names:
#     total = None
#     for ix in tqdm(range(ch.get_obj_length(obj))):
#         chunk, idcs = ch[(obj, ix)]
#         if total is None:
#             total = chunk
#         else:
#             total = clouds.merge_clouds([total, chunk])
#     objects.save2pkl(total, data_path + 'test_chunks_new/', name=obj)
