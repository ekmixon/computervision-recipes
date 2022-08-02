import os
import pytest
from crowdcounting import CrowdCountModelPose, CrowdCountModelMCNN, Router

@pytest.fixture
def local_root():
    dir_path = os.path.dirname(os.path.realpath(__file__))
    return os.path.abspath(f"{dir_path}/../..")

@pytest.fixture
def local_image_sparse(local_root):
    return os.path.join(local_root, "data/images/1.jpg")

@pytest.fixture
def local_image_dense(local_root):
    return os.path.join(local_root, "data/images/2.jpg")

@pytest.fixture
def mcnn_model(local_root):
    return os.path.join(local_root, "data/models/mcnn_shtechA_660.h5")

def test_pose_init_cpu():
    gpu_id = -1
    model = CrowdCountModelPose(gpu_id)

def test_pose_score_large_scale(local_image_sparse):
    gpu_id = -1
    model = CrowdCountModelPose(gpu_id)
    with open(local_image_sparse, 'rb') as f:
        file_bytes = f.read()
    result = model.score(file_bytes, return_image=True, img_dim=1750)    
    assert result['pred'] == 12

def test_pose_score_small_scale(local_image_sparse):
    gpu_id = -1
    model = CrowdCountModelPose(gpu_id)
    with open(local_image_sparse, 'rb') as f:
        file_bytes = f.read()
    result = model.score(file_bytes, return_image=True, img_dim=500)    

def test_mcnn_init_cpu(mcnn_model):
    gpu_id = -1
    model = CrowdCountModelMCNN(gpu_id, model_path=mcnn_model)
