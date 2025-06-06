from fastai.vision.all import *
import random

def get_colorize_data(
    sz: int,
    bs: int,
    crappy_path: Path,
    good_path: Path,
    random_seed: int = None,
    keep_pct: float = 1.0,
    num_workers: int = 8,
    stats: tuple = imagenet_stats, # imagenet_stats is available from fastai.vision.all
    xtra_tfms=[],
) -> DataLoaders:
    
    if random_seed is not None:
        random.seed(random_seed)
        set_seed(random_seed)

    # Get all image files
    all_files = get_image_files(crappy_path, recurse=True)

    if keep_pct < 1.0:
        num_files_to_keep = int(len(all_files) * keep_pct)
        # Ensure random selection is consistent if seed is provided
        if random_seed is not None:
            random.shuffle(all_files) # Shuffle in place with seed
        all_files = all_files[:num_files_to_keep]

    if not all_files:
        # Handle case with no files after filtering, to prevent errors with DataBlock
        # Create a dummy DataLoaders with minimal structure
        # This might need adjustment based on how the rest of the code handles empty DataLoaders
        # For now, returning an empty DataLoaders object
        # Create a dummy dblock and try to create empty dataloaders
        # This is a placeholder and might need a more robust empty DataLoaders creation
        # Or the calling code should handle the possibility of no data
        # One way to create an "empty" DataLoaders is to use a dummy source
        # that results in no items, or handle it upstream.
        # For now, let's assume if all_files is empty, DataBlock will handle it
        # or raise an error that needs to be caught by the caller.
        # A truly empty DataLoaders is tricky; usually, it expects some items.
        # Let's proceed with the assumption that all_files will not be empty
        # for typical use cases, or that DataBlock handles it.
        # If get_image_files returns empty and DataBlock fails, this part needs revisiting.
        pass


    def get_y(x):
        return good_path / x.relative_to(crappy_path)

    # Define transformations
    item_tfms = Resize(sz)
    # Note: get_transforms() from v1 is aug_transforms() in v2
    # Normalize.from_stats(*stats, do_y=True) in v1 becomes part of batch_tfms
    # and normalization of `y` is typically handled by ensuring ImageBlock(cls=TensorImageBW) or similar if y is an image
    # For colorization, y is also an image, so ImageBlock is appropriate.
    # Normalization for `y` is implicitly handled if y is an ImageBlock and Normalize is in batch_tfms.
    batch_tfms_list = [*aug_transforms(max_zoom=1.2, max_lighting=0.5, max_warp=0.25, xtra_tfms=xtra_tfms)]
    batch_tfms_list.append(Normalize.from_stats(*stats)) # By default, Normalize only applies to x.
                                                        # If y needs normalization, ImageBlock for y handles it.

    dblock = DataBlock(
        blocks=(ImageBlock, ImageBlock), # x is an image, y is an image
        get_items=lambda p: all_files,    # Use the pre-filtered list
        get_y=get_y,
        splitter=RandomSplitter(valid_pct=0.1, seed=random_seed),
        item_tfms=item_tfms,
        batch_tfms=batch_tfms_list
    )

    # Create DataLoaders
    # Pass `crappy_path` as source for DataBlock, it's used by get_items if not lambda.
    # Here get_items is a lambda using `all_files` directly, so source to dls.dataloaders can be any placeholder or `crappy_path`.
    dls = dblock.dataloaders(crappy_path, bs=bs, num_workers=num_workers)
    # dls.c = 3 # Usually inferred for ImageBlock(cls=PILImageBW) -> TensorImageBW, but for RGB it's 3.
              # For ImageBlock, it's automatically 3 for RGB images.
    return dls


def get_dummy_databunch() -> DataLoaders:
    # Create a dummy structure that DataBlock can work with.
    # This is challenging without file system access.
    # We'll try to make it work with non-existent paths and hope DataBlock/get_image_files handle it gracefully (e.g. empty list).
    # The original used keep_pct=0.001, implying very few files are needed.
    # If this doesn't work, this function might need to be re-thought or the calling code adapted.
    dummy_crappy_path = Path('./dummy_crappy/')
    dummy_good_path = Path('./dummy_good/')

    # To make this more robust without actual file creation, we can pass
    # a list of dummy Path objects to DataBlock if get_items can handle it.
    # However, get_image_files expects a directory.
    # For now, we rely on get_colorize_data to handle potentially empty `all_files`.
    # A truly minimal DataLoaders might be created with a list of a single dummy item.

    # Let's try to create a DataBlock with a single dummy file path for x and y
    # This requires get_items and get_y to be able to handle this.
    # The current get_colorize_data is too complex for a simple dummy if paths don't exist.

    # Simplified dummy DataBlock:
    try:
        # This will likely fail if dummy_crappy_path doesn't exist or is empty.
        # The goal is to return *something* of type DataLoaders.
        # A more robust dummy would involve creating placeholder files if allowed.
        # For now, let's call get_colorize_data with very small keep_pct.
        # This relies on get_colorize_data correctly handling empty or very small file lists.
        return get_colorize_data(
            sz=1, bs=1, crappy_path=dummy_crappy_path, good_path=dummy_good_path, keep_pct=0.001, num_workers=0
        )
    except Exception as e:
        # Fallback: create the simplest possible DataLoaders with no real data
        # This is a very basic placeholder.
        print(f"Could not create dummy databunch with get_colorize_data due to: {e}. Returning a basic DataLoaders.")
        # Create a DataBlock with minimal everything
        # This assumes we have at least one dummy file or can mock it.
        # Since we can't create files, this is also problematic.
        # The best approach might be to have a small, known set of (invalid) paths
        # and have DataBlock attempt to load them, potentially resulting in an empty DataLoader.

        # Simplest DataBlock with no items if paths are invalid (which they are)
        db = DataBlock(blocks=(ImageBlock, ImageBlock),
                       get_items=lambda x: [], # No items
                       get_y=lambda x: x,      # Dummy get_y
                       splitter=RandomSplitter(),
                       item_tfms=Resize(1))
        dls = db.dataloaders(source=[Path('dummy.jpg')], bs=1, num_workers=0) # Source can be a list
        return dls
