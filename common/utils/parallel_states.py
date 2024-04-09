import torch.distributed as dist
from typing import Optional

# parallel group define（which group current rank is belong to）
PIPELINE_MODEL_PARALLEL_GROUP = None
TENSOR_MODEL_PARALLEL_GROUP = None

SEQUENCE_MODEL_PARALLEL_GROUP = None
SEQUENCE_PARALLEL_WORLD_SIZE = None
SEQUENCE_PARALLEL_RANK = None

DATA_PARALLEL_GROUP = None
DATA_PARALLEL_GLOBAL_RANKS = None

SEQUENCE_DATA_PARALLEL_GROUP = None
SEQUENCE_DATA_PARALLEL_WORLD_SIZE = None
SEQUENCE_DATA_PARALLEL_RANK = None

MODEL_PARALLEL_GROUP = None

MPU_TENSOR_MODEL_PARALLEL_WORLD_SIZE = None
MPU_PIPELINE_MODEL_PARALLEL_WORLD_SIZE = None
MPU_TENSOR_MODEL_PARALLEL_RANK = None
MPU_PIPELINE_MODEL_PARALLEL_RANK = None

def initialize_model_parallel(
    tensor_model_parallel_size: int = 1,
    pipeline_model_parallel_size: int = 1,
    sequence_model_parallel_size: int = 1,       
) -> None:
    assert dist.is_initialized()
    world_size = dist.get_world_size()
    rank = dist.get_rank()

    assert world_size % (tensor_model_parallel_size * pipeline_model_parallel_size) == 0
    sequence_parallel_enabled = sequence_model_parallel_size > 1
    if sequence_parallel_enabled:
        if tensor_model_parallel_size != 1 or pipeline_model_parallel_size != 1:
            raise ValueError('sequence parallel can not used with other \
                             model parallel methods at the same time')
        if world_size % sequence_model_parallel_size != 0:
            raise ValueError(f'world size: {world_size} must be divisible by \
                             sequence parallel size: {sequence_model_parallel_size}')
    
    data_parallel_size = world_size // (tensor_model_parallel_size * pipeline_model_parallel_size * sequence_model_parallel_size)
    sequence_data_pallel_size = sequence_model_parallel_size * data_parallel_size

    num_tensor_parallel_groups = world_size // tensor_model_parallel_size
    num_pipeline_parallel_groups = world_size // pipeline_model_parallel_size
    num_sequence_parallel_groups = world_size // sequence_model_parallel_size
    num_sequence_data_parallel_groups = world_size // sequence_data_pallel_size
    
    # build data parallel group
    global DATA_PARALLEL_GROUP
    all_data_parallel_groups = []
    for i in range(pipeline_model_parallel_size):
        start_rank, end_rank = (_ * num_pipeline_parallel_groups for _ in [i, i+1])
        tp_or_sp_size = sequence_data_pallel_size if sequence_parallel_enabled else tensor_model_parallel_size
        for j in range(tp_or_sp_size):
            ranks = range(start_rank+j, end_rank, tp_or_sp_size)
            group = dist.new_group(ranks)
            all_data_parallel_groups.append(list(ranks))

            if rank in ranks:
                DATA_PARALLEL_GROUP = group

    # build sequence parallel group
    global SEQUENCE_MODEL_PARALLEL_GROUP
    all_sequence_model_parallel_groups = []
    for i in range(num_sequence_parallel_groups):
        ranks = range(i*sequence_model_parallel_size, (i+1)*sequence_model_parallel_size)
        group = dist.new_group(ranks)
        all_sequence_model_parallel_groups.append(list(ranks))
        if rank in ranks:
            SEQUENCE_MODEL_PARALLEL_GROUP = group

    # build sequence data parallel group
    global SEQUENCE_DATA_PARALLEL_GROUP
    if sequence_parallel_enabled:
        all_sequence_data_parallel_groups = []
        for i in range(num_sequence_data_parallel_groups):
            ranks = range(i*sequence_data_pallel_size, (i+1)*sequence_data_pallel_size)
            group = dist.new_group(ranks)
            all_sequence_data_parallel_groups.append(list(ranks))
            if rank in ranks:
                SEQUENCE_DATA_PARALLEL_GROUP = group
    else:
        SEQUENCE_DATA_PARALLEL_GROUP = DATA_PARALLEL_GROUP

    # build model parallel group
    global MODEL_PARALLEL_GROUP
    num_model_parallel_groups = sequence_data_pallel_size if sequence_parallel_enabled else data_parallel_size
    model_parallel_groups = all_sequence_data_parallel_groups if sequence_parallel_enabled else all_data_parallel_groups
    for i in range(num_model_parallel_groups):
        ranks = [model_parallel_group[i] for model_parallel_group in model_parallel_groups]
        group = dist.new_group(ranks)

        if rank in ranks:
            MODEL_PARALLEL_GROUP = group

    # build model parallel group
    global TENSOR_MODEL_PARALLEL_GROUP
    assert TENSOR_MODEL_PARALLEL_GROUP is None, 'tensor model parallel group is already initialized'
    for i in range(num_tensor_parallel_groups):
        ranks = range(i * tensor_model_parallel_size, (i + 1) * tensor_model_parallel_size)
        group = dist.new_group(ranks)
        if rank in ranks:
            TENSOR_MODEL_PARALLEL_GROUP = group

    global PIPELINE_MODEL_PARALLEL_GROUP
    assert PIPELINE_MODEL_PARALLEL_GROUP is None, 'pipeline model parallel group is already initialized'
    for i in range(num_pipeline_parallel_groups):
        ranks = range(i * pipeline_model_parallel_size, (i + 1) * pipeline_model_parallel_size)
        group = dist.new_group(ranks)
        if rank in ranks:
            PIPELINE_MODEL_PARALLEL_GROUP = group

def model_parallel_is_initialized():
    """Check if model and data parallel groups are initialized."""
    if TENSOR_MODEL_PARALLEL_GROUP is None or PIPELINE_MODEL_PARALLEL_GROUP is None or DATA_PARALLEL_GROUP is None:
        return False
    return True

def sequence_parallel_is_initialized():
    """Check if sequence and data parallel groups are initialized."""
    if SEQUENCE_MODEL_PARALLEL_GROUP is None or \
        DATA_PARALLEL_GROUP is None:
        return False
    return True

def sequence_data_parallel_is_initialized():
    """Check if sequence data parallel groups are initialized."""
    if SEQUENCE_DATA_PARALLEL_GROUP is None:
        return False
    return True

def get_model_parallel_group():
    """Get the model parallel group the caller rank belongs to."""
    assert MODEL_PARALLEL_GROUP is not None, 'model parallel group is not initialized'
    return MODEL_PARALLEL_GROUP


def get_tensor_model_parallel_group(check_initialized=True):
    """Get the tensor model parallel group the caller rank belongs to."""
    if check_initialized:
        assert TENSOR_MODEL_PARALLEL_GROUP is not None, 'tensor model parallel group is not initialized'
    return TENSOR_MODEL_PARALLEL_GROUP


def get_pipeline_model_parallel_group():
    """Get the pipeline model parallel group the caller rank belongs to."""
    assert PIPELINE_MODEL_PARALLEL_GROUP is not None, 'pipeline_model parallel group is not initialized'
    return PIPELINE_MODEL_PARALLEL_GROUP

def get_sequence_parallel_group():
    """Get the sequence parallel group the caller rank belongs to."""
    assert SEQUENCE_MODEL_PARALLEL_GROUP is not None, \
        'sequence parallel group is not initialized'
    return SEQUENCE_MODEL_PARALLEL_GROUP


def get_sequence_data_parallel_group():
    """Get the sequence parallel group the caller rank belongs to."""
    assert SEQUENCE_DATA_PARALLEL_GROUP is not None, \
        'sequence data parallel group is not initialized'
    return SEQUENCE_DATA_PARALLEL_GROUP


def get_data_parallel_group():
    """Get the data parallel group the caller rank belongs to."""
    assert DATA_PARALLEL_GROUP is not None, 'data parallel group is not initialized'
    return DATA_PARALLEL_GROUP
    
def get_sequence_parallel_rank():
    """Return my rank for the sequence parallel group."""
    global SEQUENCE_PARALLEL_RANK
    if SEQUENCE_PARALLEL_RANK is not None:
        return SEQUENCE_PARALLEL_RANK
    return dist.get_rank(group=get_sequence_parallel_group())

def get_sequence_parallel_world_size():
    """Return world size for the sequence parallel group."""
    global SEQUENCE_PARALLEL_WORLD_SIZE
    if SEQUENCE_PARALLEL_WORLD_SIZE is not None:
        return SEQUENCE_PARALLEL_WORLD_SIZE
    return dist.get_world_size(group=get_sequence_parallel_group())

def get_data_parallel_world_size():
    """Return world size for the data parallel group."""
    return dist.get_world_size(group=get_data_parallel_group())

def get_pipeline_model_parallel_world_size():
    """Return world size for the pipeline model parallel group."""
    global MPU_PIPELINE_MODEL_PARALLEL_WORLD_SIZE
    if MPU_PIPELINE_MODEL_PARALLEL_WORLD_SIZE is not None:
        return MPU_PIPELINE_MODEL_PARALLEL_WORLD_SIZE
    return dist.get_world_size(group=get_pipeline_model_parallel_group())

def get_tensor_model_parallel_world_size():
    """Return world size for the tensor model parallel group."""
    global MPU_TENSOR_MODEL_PARALLEL_WORLD_SIZE
    if MPU_TENSOR_MODEL_PARALLEL_WORLD_SIZE is not None:
        return MPU_TENSOR_MODEL_PARALLEL_WORLD_SIZE
    return dist.get_world_size(group=get_tensor_model_parallel_group())

def get_model_parallel_world_size():
    assert get_pipeline_model_parallel_world_size() == 1, "legacy get_model_parallel_world_size is only supported if PP is disabled"
    return get_tensor_model_parallel_world_size()

def get_tensor_model_parallel_rank():
    """Return my rank for the tensor model parallel group."""
    global MPU_TENSOR_MODEL_PARALLEL_RANK
    if MPU_TENSOR_MODEL_PARALLEL_RANK is not None:
        return MPU_TENSOR_MODEL_PARALLEL_RANK
    return dist.get_rank(group=get_tensor_model_parallel_group())

def get_model_parallel_rank():
    assert get_pipeline_model_parallel_world_size() == 1, "legacy get_model_parallel_rank is only supported if PP is disabled"
    return get_tensor_model_parallel_rank()