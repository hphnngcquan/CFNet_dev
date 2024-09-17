import os
def gen_offsetkitti(embedding, seq, idx):
    folder = "/media/anda/hdd2/hpnquan/OffsetKITTI/dataset/sequences/" + seq + "/velodyne_offsets/" + str(idx[0])
    if not os.path.exists(folder[:-10]):
        os.makedirs(folder[:-10])
    embedding.clone().cpu().numpy().reshape(-1).tofile(folder)