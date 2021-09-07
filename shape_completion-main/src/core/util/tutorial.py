from architecture.f2p import F2PEncoderDecoderBase
from data.sets import DatasetMenu
from data.transforms import Center
from util.strings import tutorial, banner
from util.torch.nn import PytorchNet


# ----------------------------------------------------------------------------------------------------------------------
#
# ----------------------------------------------------------------------------------------------------------------------

@tutorial
def shortcuts_tutorial():
    print("""
    Existing shortcuts are: 
    r = reconstruction
    b = batch
    v = vertex
    d = dict / dir 
    s = string or split 
    vn = vertex normals
    n_v = num vertices
    f = face / file
    n_f = num faces 
    fn = face normals 
    gt = ground truth
    tp = template
    i = index 
    p = path 
    fp = file path 
    dp = directory path 
    hp = hyper parameters 
    ds = dataset 
    You can also concatenate - gtrb = Ground Truth Reconstruction Batched  
    """)


@tutorial
def pytorch_net_tutorial():
    # What is it? PyTorchNet is a derived class of LightningModule, allowing for extended operations on it
    # Let's see some of them:
    nn = F2PEncoderDecoderBase()  # Remember that F2PEncoderDecoder is a subclass of PytorchNet
    nn.identify_system()  # Outputs the specs of the current system - Useful for identifying existing GPUs

    banner('General Net Info')
    target_input_size = ((6890, 3), (6890, 3))
    # nn.summary(x_shape=target_input_size)
    # nn.summary(batch_size=3, x_shape=target_input_size)
    print(f'On GPU = {nn.ongpu()}')  # Whether the system is on the GPU or not. Will print False
    nn.print_memory_usage(device=0)  # Print GPU 0's memory consumption
    # print(f'Output size = {nn.output_size(x_shape=target_input_size)}')
    nn.print_weights()
    # nn.visualize(x_shape=target_input_size, frmt='pdf')  # Prints PDF to current directory

    # Let's say we have some sort of nn.Module lightning:
    banner('Some lightning from the net usecase')
    import torchvision
    nn = torchvision.models.alexnet(pretrained=False)
    # We can extend it's functionality at runtime with a monkeypatch:
    py_nn = PytorchNet.monkeypatch(nn)
    nn.print_weights()
    py_nn.summary(x_shape=(3, 28, 28), batch_size=64)


@tutorial
def dataset_tutorial():
    # Use the menu to see which datasets are implemented
    print(DatasetMenu.all_sets())
    ds = DatasetMenu.order('FaustPyProj')  # This will fail if you don't have the data on disk
    # ds.validate_dataset()  # Make sure all files are available - Only run this once, to make sure.

    # For simplicity's sake, we support the old random dataloader as well:
    ldr = ds.rand_loader(num_samples=1000, transforms=[Center()], batch_size=16, n_channels=6,
                         device='cpu-single', mode='f2p')
    for point in ldr:
        print(point)
        break

    banner('The HIT')
    ds.report_index_tree()  # Take a look at how the dataset is indexed - using the hit [HierarchicalIndexTree]

    banner('Collateral Info')
    print(f'Dataset Name = {ds.name()}')
    print(f'Number of indexed files = {ds.num_indexed()}')
    print(f'Number of full shapes = {ds.num_full_shapes()}')
    print(f'Number of projections = {ds.num_projections()}')
    print(f'Required disk space in bytes = {ds.disk_space()}')
    # You can also request a summary printout with:
    ds.data_summary(with_tree=False)  # Don't print out the tree again

    # For models with a single set of faces (SMPL or SMLR for example) you can request the face set/number of vertices
    # directly:
    banner('Face Array')
    print(ds.faces())
    print(ds.num_faces())
    print(ds.num_verts())
    # You can also ask for the null-shape the dataset - with hi : [0,0...,0]
    print(ds.null_shape(n_channels=6))
    ds.plot_null_shape(strategy='spheres', with_vnormals=True)

    # Let's look at the various sampling methods available to us:
    print(ds.defined_methods())
    # We can ask for a sample of the data with this sampling method:
    banner('Data Sample')
    samp = ds.sample(num_samples=2, transforms=[Center(keys=['gt'])], n_channels=6, method='full')
    print(samp)  # Dict with gt_hi & gt
    print(ds.num_datapoints_by_method('full'))  # 100

    samp = ds.sample(num_samples=2, transforms=[Center(keys=['gt'])], n_channels=6, method='part')
    print(samp)  # Dict with gt_hi & gt & gt_mask & gt_mask
    print(ds.num_datapoints_by_method('part'))  # 1000

    samp = ds.sample(num_samples=2, transforms=[Center(keys=['gt'])], n_channels=6, method='f2p')
    print(samp)  # Dict with gt_hi & gt & gt_mask & gt_mask & tp
    print(ds.num_datapoints_by_method('f2p'))  # 10000 tuples of (gt,tp) where the subjects are the same

    # # You can also ask for a simple loader, given by the ids you'd like to see.
    # # Pass ids = None to index the entire dataset, form point_cloud = 0 to point_cloud = num_datapoints_by_method -1
    banner('Loaders')
    single_ldr = ds.loaders(s_nums=1000, s_shuffle=True, s_transform=[Center()], n_channels=6, method='f2p',
                            batch_size=3, device='cpu-single')
    for d in single_ldr:
        print(d)
        break

    print(single_ldr.num_verts())
    # There are also operations defined on the loaders themselves. See utils.torch_data for details

    # To receive train/validation splits or train/validation/test splits use:
    my_loaders = ds.loaders(split=[0.8, 0.1, 0.1], s_nums=[2000, 1000, 1000],
                            s_shuffle=[True] * 3, s_transform=[Center()] * 3, global_shuffle=True, method='p2p',
                            s_dynamic=[True, False, False])

    # Please read the documentation of split_loaders for the exact details. In essence:
    # You'll receive len(split) dataloaders, where each part i is split[i]*num_point_clouds size. From this split,
    # s_nums[i] will be taken for the dataloader, and transformed by s_transform[i].
    # s_shuffle and global_shuffle controls the shuffling of the different partitions - see doc inside function
